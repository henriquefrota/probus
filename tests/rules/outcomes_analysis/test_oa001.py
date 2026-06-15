from pathlib import Path

import pytest

from probus.rules.base import Finding
from probus.rules.outcomes_analysis.oa001_no_transaction_costs import (
    OA001NoTransactionCosts,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def rule() -> OA001NoTransactionCosts:
    return OA001NoTransactionCosts()


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestOA001BadFixture:
    def test_flags_at_least_one_finding(self, rule):
        findings = rule.check(_load("oa001_bad.py"), "oa001_bad.py")
        assert len(findings) >= 1, "Expected OA001 to flag the bad fixture"

    def test_finding_rule_id(self, rule):
        findings = rule.check(_load("oa001_bad.py"), "oa001_bad.py")
        assert findings[0].rule_id == "OA001"

    def test_finding_severity_is_medium(self, rule):
        findings = rule.check(_load("oa001_bad.py"), "oa001_bad.py")
        assert findings[0].severity == "medium"

    def test_finding_line_is_one(self, rule):
        findings = rule.check(_load("oa001_bad.py"), "oa001_bad.py")
        assert findings[0].line == 1

    def test_finding_message_mentions_cost(self, rule):
        findings = rule.check(_load("oa001_bad.py"), "oa001_bad.py")
        msg = findings[0].message.lower()
        assert any(t in msg for t in ("cost", "commission", "transaction", "slippage"))

    def test_finding_has_recommendation(self, rule):
        findings = rule.check(_load("oa001_bad.py"), "oa001_bad.py")
        assert len(findings[0].recommendation) > 0

    def test_finding_has_reference(self, rule):
        findings = rule.check(_load("oa001_bad.py"), "oa001_bad.py")
        assert len(findings[0].reference) > 0

    def test_finding_file_matches_input(self, rule):
        findings = rule.check(_load("oa001_bad.py"), "oa001_bad.py")
        assert findings[0].file == "oa001_bad.py"

    def test_finding_is_dataclass(self, rule):
        findings = rule.check(_load("oa001_bad.py"), "oa001_bad.py")
        assert isinstance(findings[0], Finding)


class TestOA001GoodFixture:
    def test_produces_zero_findings(self, rule):
        findings = rule.check(_load("oa001_good.py"), "oa001_good.py")
        assert len(findings) == 0, (
            f"Expected zero OA001 findings on good fixture, got {len(findings)}:\n"
            + "\n".join(f"  line {f.line}: {f.message}" for f in findings)
        )


class TestOA001EdgeCases:
    def test_returns_empty_on_syntax_error(self, rule):
        assert rule.check("def (bad syntax", "x.py") == []

    def test_cost_mention_in_comment_suppresses_finding(self, rule):
        source = (
            "import numpy as np\n"
            "# ignoring transaction costs for now\n"
            "strategy_returns = signal.shift(1) * returns\n"
            "sharpe = strategy_returns.mean() / strategy_returns.std() * 252\n"
        )
        assert rule.check(source, "cost_in_comment.py") == []

    def test_slippage_mention_suppresses_finding(self, rule):
        source = (
            "slippage = 0.0005\n"
            "strategy_returns = signal.shift(1) * returns\n"
            "sharpe = strategy_returns.mean() / strategy_returns.std() * 252\n"
        )
        assert rule.check(source, "with_slippage.py") == []

    def test_no_strategy_returns_no_finding(self, rule):
        # Only a preprocessing script — no strategy return variable
        source = (
            "import pandas as pd\n"
            "returns = prices.pct_change()\n"
            "vol = returns.rolling(20).std()\n"
            "sharpe = returns.mean() / returns.std() * 252\n"
        )
        assert rule.check(source, "preprocess.py") == []

    def test_no_performance_eval_no_finding(self, rule):
        # Has strategy_returns but no Sharpe/annualization
        source = (
            "strategy_returns = signal * returns\n" "print(strategy_returns.sum())\n"
        )
        assert rule.check(source, "partial.py") == []

    def test_test_file_excluded(self, rule):
        source = (
            "strategy_returns = signal.shift(1) * returns\n"
            "sharpe = strategy_returns.mean() / strategy_returns.std() * 252\n"
        )
        assert rule.check(source, "tests/test_backtest.py") == []

    def test_daily_pnl_triggers_condition1(self, rule):
        source = (
            "daily_pnl = signal * returns\n"
            "sharpe = daily_pnl.mean() / daily_pnl.std() * 252\n"
        )
        findings = rule.check(source, "pnl_backtest.py")
        assert len(findings) >= 1
        assert findings[0].rule_id == "OA001"

    def test_cumprod_triggers_condition1(self, rule):
        source = (
            "strategy_returns = signal.shift(1) * returns\n"
            "cumulative = (1 + strategy_returns).cumprod()\n"
            "sharpe = strategy_returns.mean() / strategy_returns.std() * 252\n"
        )
        findings = rule.check(source, "cumprod_backtest.py")
        assert len(findings) >= 1

    def test_fee_mention_suppresses_finding(self, rule):
        source = (
            "fee = 0.002\n"
            "strategy_returns = signal.shift(1) * returns - fee\n"
            "sharpe = strategy_returns.mean() / strategy_returns.std() * 252\n"
        )
        assert rule.check(source, "with_fee.py") == []
