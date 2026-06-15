"""
CS001 — LOOKAHEAD_SHIFT_NEGATIVE

Detects calls to .shift() with a negative integer argument on DataFrame
columns or Series that appear to feed into a signal or model computation.

WHY THIS MATTERS
----------------
In pandas, df['col'].shift(-N) moves each value N rows backward, meaning
the value at index t receives the observation from t+N. In a time-ordered
dataset this is future data: a model that uses shift(-1) on a feature is
reading tomorrow's value to make today's decision. This is look-ahead bias.

The effect on backtests is systematic inflation of performance metrics. The
strategy appears to predict the market because it literally contains the
outcome it is pretending to forecast.

LEGITIMATE EXCEPTION
--------------------
shift(-N) is correct when creating a supervised learning target — a column
that holds the *future return we want to predict*, not a feature used to
generate the signal. Common forms:

    target = df['close'].pct_change().shift(-1)
    fwd_return = prices.shift(-1) / prices - 1

Probus excludes shift(-N) calls that are directly assigned to a variable
whose name clearly signals a label or forward-return intent:
  - Exact names: target, y, label, labels, fwd_return, fwd_ret, fwd_returns,
    forward_return, forward_returns, next_return, future_return, future_returns
  - Prefixes: target_, y_, label_, fwd_, forward_, next_, future_
  - Suffixes: _target, _label, _fwd, _forward

DETECTION LOGIC
---------------
1. Collect all .shift(N) calls where N is a constant negative integer.
2. Collect the subset of those calls that occur inside an assignment to a
   label-like variable (safe set).
3. Flag every call in step 1 that is not in the safe set.

KNOWN LIMITATIONS
-----------------
- Variable-argument shifts like .shift(-n) where n is a runtime value are
  not detected (cannot determine sign statically).
- Cross-function and cross-cell tracking is not performed.
- A label-like variable that also feeds a signal pipeline would be excluded
  from this check despite being problematic.
- Test files are excluded.

REFERENCES
----------
- Federal Reserve SR 11-7, §Conceptual Soundness (2011)
- BCB Resolução 4557 (2017)
- López de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 7.
"""

import ast
from typing import Optional

from probus.rules._utils import is_test_filepath
from probus.rules.base import Finding, Rule

# Variable names that indicate the assignment is a supervised learning label,
# not a signal. shift(-N) inside these assignments is intentional.
_LABEL_NAMES: frozenset[str] = frozenset(
    {
        "target",
        "y",
        "label",
        "labels",
        "fwd_return",
        "fwd_ret",
        "fwd_returns",
        "forward_return",
        "forward_returns",
        "next_return",
        "future_return",
        "future_returns",
    }
)
_LABEL_PREFIXES: tuple[str, ...] = (
    "target_",
    "y_",
    "label_",
    "fwd_",
    "forward_",
    "next_",
    "future_",
)
_LABEL_SUFFIXES: tuple[str, ...] = ("_target", "_label", "_fwd", "_forward")


def _is_label_like(name: str) -> bool:
    lower = name.lower()
    return (
        lower in _LABEL_NAMES
        or any(lower.startswith(p) for p in _LABEL_PREFIXES)
        or any(lower.endswith(s) for s in _LABEL_SUFFIXES)
    )


def _get_negative_shift_arg(node: ast.Call) -> Optional[int]:
    """
    If node is a .shift() call whose first positional argument is a constant
    negative integer, return that integer. Otherwise return None.

    Handles both representations Python may produce:
      shift(-1) → UnaryOp(USub(), Constant(1))   (most common)
      shift(-1) → Constant(-1)                    (constant-folded, rare)
    """
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "shift"):
        return None
    if not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
        if isinstance(arg.operand, ast.Constant) and isinstance(arg.operand.value, int):
            return -arg.operand.value
    if isinstance(arg, ast.Constant) and isinstance(arg.value, int) and arg.value < 0:
        return arg.value
    return None


def _collect_negative_shifts(tree: ast.AST) -> list[tuple[int, int]]:
    """
    Return (line_number, shift_value) for every .shift(N<0) call in the file.
    shift_value is negative (e.g. -1 for shift(-1)).
    """
    result: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            neg = _get_negative_shift_arg(node)
            if neg is not None:
                result.append((node.lineno, neg))
    return result


def _collect_safe_shift_lines(tree: ast.AST) -> set[int]:
    """
    Return line numbers of shift(-N) calls that are part of a label/target
    variable assignment. These are legitimate supervised learning patterns
    and must not be flagged.
    """
    safe: set[int] = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        # Determine whether any target of this assignment is label-like.
        is_label = False
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and _is_label_like(target.id):
                is_label = True
                break
            # df['target'] = ...  or  df['fwd_return'] = ...
            if isinstance(target, ast.Subscript):
                key = target.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if _is_label_like(key.value):
                        is_label = True
                        break

        if not is_label:
            continue

        value = node.value if isinstance(node, ast.Assign) else node.value
        if value is None:
            continue

        # Mark every shift(-N) call found anywhere in the RHS as safe.
        for subnode in ast.walk(value):
            if isinstance(subnode, ast.Call):
                neg = _get_negative_shift_arg(subnode)
                if neg is not None:
                    safe.add(subnode.lineno)

    return safe


class CS001LookaheadShiftNegative(Rule):
    """
    CS001 — LOOKAHEAD_SHIFT_NEGATIVE

    Detects .shift(N) calls where N is a negative integer, which access
    future observations and constitute look-ahead bias.

    Bad example::

        signal = df['close'].rolling(20).mean().shift(-1)   # reads tomorrow's MA

    Good example::

        signal = df['close'].rolling(20).mean()             # current-period MA
        # shift(-1) is fine only when building a supervised learning target:
        target = df['close'].pct_change().shift(-1)
    """

    rule_id = "CS001"
    severity = "critical"

    def check(self, source: str, filepath: str) -> list[Finding]:
        if is_test_filepath(filepath):
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        negative_shifts = _collect_negative_shifts(tree)
        if not negative_shifts:
            return []

        safe_lines = _collect_safe_shift_lines(tree)
        findings: list[Finding] = []

        for line, value in negative_shifts:
            if line in safe_lines:
                continue
            findings.append(
                Finding(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    message=(
                        f"shift({value}) at line {line} shifts data backward by "
                        f"{abs(value)} period(s), reading future observations into "
                        "the current index position. Any signal or feature derived "
                        "from this call has look-ahead bias."
                    ),
                    line=line,
                    file=filepath,
                    recommendation=(
                        "Remove the negative shift from signal and feature construction. "
                        "If you need an execution lag, use shift(1) on the signal side. "
                        "If this shift(-N) is creating a supervised learning target "
                        "(forward return), assign the result to a variable named "
                        "'target', 'y', 'fwd_return', or similar — those assignments "
                        "are excluded from this check."
                    ),
                    reference=(
                        "SR 11-7 §Conceptual Soundness; "
                        "López de Prado (2018) Advances in Financial Machine Learning, Ch. 7"
                    ),
                )
            )

        return findings
