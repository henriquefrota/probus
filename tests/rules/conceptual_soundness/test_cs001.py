from pathlib import Path

import pytest

from probus.rules.base import Finding
from probus.rules.conceptual_soundness.cs001_lookahead_shift_negative import (
    CS001LookaheadShiftNegative,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def rule() -> CS001LookaheadShiftNegative:
    return CS001LookaheadShiftNegative()


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestCS001BadFixture:
    def test_flags_at_least_one_finding(self, rule):
        findings = rule.check(_load("cs001_bad.py"), "cs001_bad.py")
        assert len(findings) >= 1, "Expected CS001 to flag the bad fixture"

    def test_finding_rule_id(self, rule):
        findings = rule.check(_load("cs001_bad.py"), "cs001_bad.py")
        assert findings[0].rule_id == "CS001"

    def test_finding_severity_is_critical(self, rule):
        findings = rule.check(_load("cs001_bad.py"), "cs001_bad.py")
        assert findings[0].severity == "critical"

    def test_finding_line_is_positive(self, rule):
        findings = rule.check(_load("cs001_bad.py"), "cs001_bad.py")
        assert isinstance(findings[0].line, int) and findings[0].line > 0

    def test_finding_line_contains_shift(self, rule):
        findings = rule.check(_load("cs001_bad.py"), "cs001_bad.py")
        source_lines = _load("cs001_bad.py").splitlines()
        flagged_line = source_lines[findings[0].line - 1]
        assert (
            "shift" in flagged_line
        ), f"Line {findings[0].line} does not contain 'shift': {flagged_line!r}"

    def test_finding_message_mentions_shift(self, rule):
        findings = rule.check(_load("cs001_bad.py"), "cs001_bad.py")
        assert "shift" in findings[0].message.lower()

    def test_finding_has_recommendation(self, rule):
        findings = rule.check(_load("cs001_bad.py"), "cs001_bad.py")
        assert len(findings[0].recommendation) > 0

    def test_finding_has_reference(self, rule):
        findings = rule.check(_load("cs001_bad.py"), "cs001_bad.py")
        assert len(findings[0].reference) > 0

    def test_finding_file_matches_input(self, rule):
        findings = rule.check(_load("cs001_bad.py"), "cs001_bad.py")
        assert findings[0].file == "cs001_bad.py"

    def test_finding_is_dataclass(self, rule):
        findings = rule.check(_load("cs001_bad.py"), "cs001_bad.py")
        assert isinstance(findings[0], Finding)


class TestCS001GoodFixture:
    def test_produces_zero_findings(self, rule):
        findings = rule.check(_load("cs001_good.py"), "cs001_good.py")
        assert len(findings) == 0, (
            f"Expected zero CS001 findings on good fixture, got {len(findings)}:\n"
            + "\n".join(f"  line {f.line}: {f.message}" for f in findings)
        )


class TestCS001EdgeCases:
    def test_returns_empty_on_syntax_error(self, rule):
        assert rule.check("def (bad syntax", "x.py") == []

    def test_no_shift_produces_no_findings(self, rule):
        source = "import pandas as pd\ndf['ma'] = df['close'].rolling(20).mean()\n"
        assert rule.check(source, "x.py") == []

    def test_positive_shift_not_flagged(self, rule):
        source = "import pandas as pd\ndf['signal'] = df['ma'].shift(1)\n"
        assert rule.check(source, "x.py") == []

    def test_zero_shift_not_flagged(self, rule):
        source = "import pandas as pd\ndf['x'] = df['close'].shift(0)\n"
        assert rule.check(source, "x.py") == []

    def test_target_assignment_excluded(self, rule):
        source = "target = df['close'].pct_change().shift(-1)\n"
        assert rule.check(source, "x.py") == []

    def test_fwd_return_assignment_excluded(self, rule):
        source = "fwd_return = df['close'].pct_change().shift(-1)\n"
        assert rule.check(source, "x.py") == []

    def test_y_assignment_excluded(self, rule):
        source = "y = prices.shift(-1) / prices - 1\n"
        assert rule.check(source, "x.py") == []

    def test_label_column_assignment_excluded(self, rule):
        # df['target'] = df['close'].shift(-1)
        source = "df['target'] = df['close'].shift(-1)\n"
        assert rule.check(source, "x.py") == []

    def test_non_label_variable_is_flagged(self, rule):
        source = "signal = df['close'].rolling(5).mean().shift(-1)\n"
        findings = rule.check(source, "x.py")
        assert len(findings) >= 1
        assert findings[0].rule_id == "CS001"

    def test_variable_shift_arg_not_flagged(self, rule):
        # shift(-n) where n is a variable — static analysis cannot determine sign
        source = "n = 1\ndf['x'] = df['close'].shift(-n)\n"
        assert rule.check(source, "x.py") == []
