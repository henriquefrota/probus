from pathlib import Path

import pytest

from probus.rules.base import Finding
from probus.rules.conceptual_soundness.cs002_same_period_signal_execution import (
    CS002SamePeriodSignalExecution,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def rule() -> CS002SamePeriodSignalExecution:
    return CS002SamePeriodSignalExecution()


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestCS002BadFixture:
    def test_flags_at_least_one_finding(self, rule):
        findings = rule.check(_load("cs002_bad.py"), "cs002_bad.py")
        assert len(findings) >= 1, "Expected CS002 to flag the bad fixture"

    def test_finding_rule_id(self, rule):
        findings = rule.check(_load("cs002_bad.py"), "cs002_bad.py")
        assert findings[0].rule_id == "CS002"

    def test_finding_severity_is_high(self, rule):
        findings = rule.check(_load("cs002_bad.py"), "cs002_bad.py")
        assert findings[0].severity == "high"

    def test_finding_line_is_positive(self, rule):
        findings = rule.check(_load("cs002_bad.py"), "cs002_bad.py")
        assert isinstance(findings[0].line, int) and findings[0].line > 0

    def test_finding_line_contains_multiplication(self, rule):
        findings = rule.check(_load("cs002_bad.py"), "cs002_bad.py")
        source_lines = _load("cs002_bad.py").splitlines()
        flagged_line = source_lines[findings[0].line - 1]
        assert "*" in flagged_line, (
            f"Line {findings[0].line} does not contain '*': {flagged_line!r}"
        )

    def test_finding_message_mentions_shift(self, rule):
        findings = rule.check(_load("cs002_bad.py"), "cs002_bad.py")
        assert "shift" in findings[0].message.lower() or "bar" in findings[0].message.lower()

    def test_finding_has_recommendation(self, rule):
        findings = rule.check(_load("cs002_bad.py"), "cs002_bad.py")
        assert len(findings[0].recommendation) > 0

    def test_finding_has_reference(self, rule):
        findings = rule.check(_load("cs002_bad.py"), "cs002_bad.py")
        assert len(findings[0].reference) > 0

    def test_finding_file_matches_input(self, rule):
        findings = rule.check(_load("cs002_bad.py"), "cs002_bad.py")
        assert findings[0].file == "cs002_bad.py"

    def test_finding_is_dataclass(self, rule):
        findings = rule.check(_load("cs002_bad.py"), "cs002_bad.py")
        assert isinstance(findings[0], Finding)


class TestCS002GoodFixture:
    def test_produces_zero_findings(self, rule):
        findings = rule.check(_load("cs002_good.py"), "cs002_good.py")
        assert len(findings) == 0, (
            f"Expected zero CS002 findings on good fixture, got {len(findings)}:\n"
            + "\n".join(f"  line {f.line}: {f.message}" for f in findings)
        )


class TestCS002EdgeCases:
    def test_returns_empty_on_syntax_error(self, rule):
        assert rule.check("def (bad syntax", "x.py") == []

    def test_no_multiplication_produces_no_findings(self, rule):
        source = (
            "returns = prices.pct_change()\n"
            "signal = (ma > mb).astype(int)\n"
        )
        assert rule.check(source, "x.py") == []

    def test_shift1_on_signal_not_flagged(self, rule):
        source = (
            "returns = prices.pct_change()\n"
            "strategy = signal.shift(1) * returns\n"
        )
        assert rule.check(source, "x.py") == []

    def test_shift1_keyword_not_flagged(self, rule):
        source = (
            "returns = prices.pct_change()\n"
            "strategy = signal.shift(periods=1) * returns\n"
        )
        assert rule.check(source, "x.py") == []

    def test_both_sides_return_like_not_flagged(self, rule):
        # Excess return vs benchmark — both sides are returns.
        source = (
            "active_return = portfolio_returns * benchmark_returns\n"
        )
        assert rule.check(source, "x.py") == []

    def test_inline_pct_change_flagged(self, rule):
        # pct_change() used directly in the multiplication (no variable).
        source = "strategy = signal * prices.pct_change()\n"
        findings = rule.check(source, "x.py")
        assert len(findings) >= 1
        assert findings[0].rule_id == "CS002"

    def test_inline_pct_change_with_shift1_not_flagged(self, rule):
        source = "strategy = signal.shift(1) * prices.pct_change()\n"
        assert rule.check(source, "x.py") == []

    def test_returns_on_left_flagged(self, rule):
        # Return on the left, unshifted signal on the right.
        source = (
            "returns = prices.pct_change()\n"
            "strategy = returns * signal\n"
        )
        findings = rule.check(source, "x.py")
        assert len(findings) >= 1
        assert findings[0].rule_id == "CS002"
