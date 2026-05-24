"""
OA001 — NO_TRANSACTION_COSTS

Detects complete backtests that compute and report strategy performance
without any reference to transaction costs.

WHY THIS MATTERS
----------------
Transaction costs (commissions, bid-ask spread, market impact, slippage)
are the primary mechanism through which theoretical backtest performance
degrades in live trading. A strategy with a gross Sharpe of 1.2 may have
a net Sharpe of 0.3 after realistic cost assumptions are applied.

SR 11-7 §Outcomes Analysis requires that model outputs be evaluated against
realistic assumptions. A backtest that ignores transaction costs is not a
valid model output — it is an upper bound on performance under zero-friction
assumptions that never exist in practice.

DETECTION LOGIC
---------------
Flags when ALL THREE conditions are simultaneously true:

  Condition 1 — Strategy return calculation present:
    Any of: assignment to a variable named strategy_returns, portfolio_returns,
    pnl, daily_pnl, or similar; or a .cumprod() / .cumsum() call.

  Condition 2 — Performance evaluation present:
    Any of: multiplication by 252 (annualization); a variable named
    sharpe (case-insensitive); or both .mean() and .std() called
    (Sharpe-ratio pattern).

  Condition 3 — No cost mention anywhere in the file:
    The source contains none of: commission, cost, fee, spread, slippage,
    bps, transaction, rebate (case-insensitive, full-text search including
    variable names, comments, and strings).

Benefit of the doubt: if any cost-related term appears anywhere (even in a
comment like "# ignoring transaction costs for now"), the rule does not flag.

CONSERVATISM NOTE
-----------------
The rule requires all three conditions simultaneously. A file with no strategy
returns (e.g., a preprocessing script) is never flagged. A file that merely
calls .pct_change() and .mean() without a performance calculation is not
flagged. The combination ensures the rule targets complete, self-contained
backtests rather than analytical snippets.

KNOWN LIMITATIONS
-----------------
- Transaction costs implemented as an external function call in a separate
  module will not be detected (cross-module analysis is not performed).
- Cost variable names using non-standard terminology (e.g., "rake", "haircut")
  are not detected.
- Test files are excluded.

REFERENCES
----------
- Federal Reserve SR 11-7, §Outcomes Analysis (2011)
- BCB Resolução 4557 (2017)
- López de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 8.
"""

import ast

from probus.rules.base import Finding, Rule

_STRATEGY_VAR_NAMES: frozenset[str] = frozenset({
    "strategy_returns",
    "portfolio_returns",
    "pnl",
    "daily_pnl",
    "strategy_pnl",
    "port_returns",
    "strat_returns",
    "net_returns",
    "gross_returns",
})

_COST_TERMS: frozenset[str] = frozenset({
    "commission",
    "cost",
    "fee",
    "spread",
    "slippage",
    "bps",
    "transaction",
    "rebate",
})


def _is_test_filepath(filepath: str) -> bool:
    normalized = filepath.replace("\\", "/")
    return (
        "/tests/" in normalized
        or "/test_" in normalized
        or normalized.startswith("test_")
    )


def _has_strategy_returns(tree: ast.AST) -> bool:
    """
    Condition 1: file contains a strategy return calculation.

    Detected as:
      - Assignment to a strategy-like variable name
      - A .cumprod() or .cumsum() call (cumulative return series)
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _STRATEGY_VAR_NAMES:
                    return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ("cumprod", "cumsum"):
                    return True
    return False


def _has_performance_evaluation(tree: ast.AST) -> bool:
    """
    Condition 2: file contains a performance metric calculation.

    Detected as:
      - Multiplication by 252 (annualization factor)
      - A variable named 'sharpe' (case-insensitive)
      - Both .mean() and .std() called (Sharpe ratio pattern)
    """
    method_calls: set[str] = set()
    for node in ast.walk(tree):
        # Multiplication by 252
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            for operand in (node.left, node.right):
                if isinstance(operand, ast.Constant) and operand.value == 252:
                    return True

        # Variable named 'sharpe'
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and "sharpe" in target.id.lower():
                    return True

        # Collect method call names
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method_calls.add(node.func.attr)

    # Both mean and std present → Sharpe-ratio pattern
    return "mean" in method_calls and "std" in method_calls


def _has_cost_mention(source: str) -> bool:
    """Condition 3 (inverted): any cost-related term anywhere in the source."""
    lower = source.lower()
    return any(term in lower for term in _COST_TERMS)


class OA001NoTransactionCosts(Rule):
    """
    OA001 — NO_TRANSACTION_COSTS

    Detects backtests that compute and report strategy performance without
    any mention of transaction costs.

    Bad example::

        strategy_returns = signal.shift(1) * returns
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        print(f"Sharpe: {sharpe:.2f}")

    Good example::

        commission = 0.001
        strategy_returns = signal.shift(1) * returns - commission * signal.diff().abs()
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        print(f"Sharpe: {sharpe:.2f}")
    """

    rule_id = "OA001"
    severity = "medium"

    def check(self, source: str, filepath: str) -> list[Finding]:
        if _is_test_filepath(filepath):
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        if not _has_strategy_returns(tree):
            return []

        if not _has_performance_evaluation(tree):
            return []

        if _has_cost_mention(source):
            return []

        return [
            Finding(
                rule_id=self.rule_id,
                severity=self.severity,
                message=(
                    "No mention of transaction costs (commission, fee, spread, "
                    "slippage, bps) found anywhere in this file. This backtest "
                    "appears to calculate strategy returns and evaluate performance "
                    "without accounting for trading costs, which typically inflates "
                    "Sharpe ratios and cumulative returns. (File-level check — "
                    "Line 1 is reported by convention.)"
                ),
                line=1,
                file=filepath,
                recommendation=(
                    "Add realistic transaction cost assumptions: at minimum, a "
                    "round-trip commission per trade and an estimate of bid-ask "
                    "spread. Apply costs to each signal change: "
                    "net_returns = gross_returns - cost_per_trade * turnover. "
                    "For equity strategies, 5-20 bps round-trip is a common "
                    "starting assumption; validate against your actual execution data."
                ),
                reference=(
                    "SR 11-7 §Outcomes Analysis; "
                    "López de Prado (2018) Advances in Financial Machine Learning, Ch. 8"
                ),
            )
        ]
