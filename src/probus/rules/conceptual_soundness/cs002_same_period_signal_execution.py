"""
CS002 — SAME_PERIOD_SIGNAL_EXECUTION

Detects when a trading signal and period returns are combined arithmetically
without the one-period execution lag required by realistic backtesting.

WHY THIS MATTERS
----------------
A signal generated at the close of day t (using day t's prices) can only be
executed at the earliest at the open of day t+1. Multiplying that signal
directly by day t's return — which measures price movement from the close
of t-1 to the close of t — implies the strategy entered the position at
the open of t-1 and exited at the close of t, before the signal was even
computed. This is same-bar execution, a form of look-ahead bias.

The correct pattern shifts the signal by one period before weighting returns:

    WRONG:  strategy = signal * returns         # signal and return same period
    CORRECT: strategy = signal.shift(1) * returns  # signal executed next bar

DETECTION LOGIC
---------------
Flags a direct multiplication A * B in an assignment when:
  1. Exactly one operand is return-like:
       - A direct .pct_change() or .diff() call
       - A variable tracked as holding pct_change/diff output
       - A subscript whose column name contains "return", "ret", or "pct"
  2. The other operand (the signal) is NOT wrapped in .shift(1).

"Exactly one return-like operand" avoids flagging legitimate portfolio
arithmetic where both sides are returns (e.g., excess return vs benchmark).

CONSERVATISM NOTE
-----------------
This rule only inspects top-level multiplications in assignment statements.
It does not perform data-flow analysis across functions or files. If the
signal has already been shifted in a previous statement and is held in a
variable, this rule will not catch the same-bar execution error but also
will not produce a false positive on that pattern — the shift happens to
be applied before the multiplication site.

KNOWN LIMITATIONS
-----------------
- Chained multiplications (a * b * c) are inspected at the outermost level
  only. The inner pair is not directly checked.
- .multiply() method calls are not yet detected (only * operator).
- Cross-function analysis is not performed.

REFERENCES
----------
- Federal Reserve SR 11-7, §Conceptual Soundness (2011)
- BCB Resolução 4557 (2017)
- López de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 8.
"""

import ast

from probus.rules.base import Finding, Rule

# pandas methods that produce period-over-period returns.
_RETURN_METHODS: frozenset[str] = frozenset({"pct_change", "diff"})


def _is_return_expr(node: ast.AST, return_vars: set[str]) -> bool:
    """
    Return True if node clearly represents period returns.

    Three forms are recognized:
      - Direct call:  df['close'].pct_change()
      - Tracked var:  returns  (previously assigned from pct_change/diff)
      - Return-named subscript:  df['returns'], df['ret'], df['pct']
    """
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr in _RETURN_METHODS:
            return True
    if isinstance(node, ast.Name) and node.id in return_vars:
        return True
    if isinstance(node, ast.Subscript):
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            lower = key.value.lower()
            if "return" in lower or lower in {"ret", "rets", "pct"}:
                return True
    return False


def _is_shift1(node: ast.AST) -> bool:
    """
    Return True if node is an expression wrapped in .shift(1).

    Accepts both positional and keyword forms:
      signal.shift(1)
      signal.shift(periods=1)
    """
    if not isinstance(node, ast.Call):
        return False
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "shift"):
        return False
    if node.args:
        if isinstance(node.args[0], ast.Constant) and node.args[0].value == 1:
            return True
    for kw in node.keywords:
        if kw.arg == "periods" and isinstance(kw.value, ast.Constant) and kw.value.value == 1:
            return True
    return False


def _collect_return_vars(tree: ast.AST) -> set[str]:
    """
    Collect variable names that are directly assigned the result of a
    pct_change() or diff() expression (including chains like .fillna(0)).
    """
    result: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        has_return_call = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in _RETURN_METHODS
            for n in ast.walk(node.value)
        )
        if has_return_call:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result.add(target.id)
    return result


class CS002SamePeriodSignalExecution(Rule):
    """
    CS002 — SAME_PERIOD_SIGNAL_EXECUTION

    Detects when period returns and a signal are multiplied without the
    signal being shifted by one period first.

    Bad example::

        returns = df['close'].pct_change()
        signal  = (ma_fast > ma_slow).astype(int)
        strategy = signal * returns          # same-bar execution

    Good example::

        returns = df['close'].pct_change()
        signal  = (ma_fast > ma_slow).astype(int)
        strategy = signal.shift(1) * returns  # one-period lag
    """

    rule_id = "CS002"
    severity = "high"

    def check(self, source: str, filepath: str) -> list[Finding]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        return_vars = _collect_return_vars(tree)
        findings: list[Finding] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Mult):
                continue

            left, right = value.left, value.right
            left_is_ret = _is_return_expr(left, return_vars)
            right_is_ret = _is_return_expr(right, return_vars)

            # Only flag when exactly one side is return-like.
            # If both are returns (or neither is), the pattern is ambiguous.
            if left_is_ret == right_is_ret:
                continue

            signal_node = right if left_is_ret else left

            if not _is_shift1(signal_node):
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=self.severity,
                        message=(
                            f"At line {node.lineno}, a signal is multiplied with period "
                            "returns without an intervening shift(1). This implies the "
                            "signal was generated and acted upon within the same bar, "
                            "which is impossible in live trading."
                        ),
                        line=node.lineno,
                        file=filepath,
                        recommendation=(
                            "Apply signal.shift(1) before multiplying with returns: "
                            "strategy = signal.shift(1) * returns. "
                            "The signal computed at the close of day t should be "
                            "executed at the earliest at the open of day t+1."
                        ),
                        reference=(
                            "SR 11-7 §Conceptual Soundness; "
                            "López de Prado (2018) Advances in Financial Machine Learning, Ch. 8"
                        ),
                    )
                )

        return findings
