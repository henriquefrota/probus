from pathlib import Path

import pytest

from probus.rules.base import Finding
from probus.rules.robustness_testing.rt001_random_split_time_series import (
    RT001RandomSplitTimeSeries,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def rule() -> RT001RandomSplitTimeSeries:
    return RT001RandomSplitTimeSeries()


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestRT001BadFixture:
    def test_flags_at_least_one_finding(self, rule):
        findings = rule.check(_load("rt001_bad.py"), "rt001_bad.py")
        assert len(findings) >= 1, "Expected RT001 to flag the bad fixture"

    def test_finding_rule_id(self, rule):
        findings = rule.check(_load("rt001_bad.py"), "rt001_bad.py")
        assert findings[0].rule_id == "RT001"

    def test_finding_severity_is_medium(self, rule):
        findings = rule.check(_load("rt001_bad.py"), "rt001_bad.py")
        assert findings[0].severity == "medium"

    def test_finding_line_is_positive(self, rule):
        findings = rule.check(_load("rt001_bad.py"), "rt001_bad.py")
        assert isinstance(findings[0].line, int) and findings[0].line > 0

    def test_finding_line_contains_train_test_split(self, rule):
        findings = rule.check(_load("rt001_bad.py"), "rt001_bad.py")
        source_lines = _load("rt001_bad.py").splitlines()
        flagged = source_lines[findings[0].line - 1]
        assert "train_test_split" in flagged, (
            f"Line {findings[0].line} does not contain train_test_split: {flagged!r}"
        )

    def test_finding_message_mentions_shuffle(self, rule):
        findings = rule.check(_load("rt001_bad.py"), "rt001_bad.py")
        assert "shuffle" in findings[0].message.lower()

    def test_finding_has_recommendation(self, rule):
        findings = rule.check(_load("rt001_bad.py"), "rt001_bad.py")
        assert len(findings[0].recommendation) > 0

    def test_finding_has_reference(self, rule):
        findings = rule.check(_load("rt001_bad.py"), "rt001_bad.py")
        assert len(findings[0].reference) > 0

    def test_finding_file_matches_input(self, rule):
        findings = rule.check(_load("rt001_bad.py"), "rt001_bad.py")
        assert findings[0].file == "rt001_bad.py"

    def test_finding_is_dataclass(self, rule):
        findings = rule.check(_load("rt001_bad.py"), "rt001_bad.py")
        assert isinstance(findings[0], Finding)


class TestRT001GoodFixture:
    def test_produces_zero_findings(self, rule):
        findings = rule.check(_load("rt001_good.py"), "rt001_good.py")
        assert len(findings) == 0, (
            f"Expected zero RT001 findings on good fixture, got {len(findings)}:\n"
            + "\n".join(f"  line {f.line}: {f.message}" for f in findings)
        )


class TestRT001EdgeCases:
    def test_returns_empty_on_syntax_error(self, rule):
        assert rule.check("def (bad syntax", "x.py") == []

    def test_no_train_test_split_produces_no_findings(self, rule):
        source = (
            "import pandas as pd\n"
            "dates = pd.date_range('2020-01-01', periods=100, freq='B')\n"
            "returns = prices.pct_change()\n"
        )
        assert rule.check(source, "no_split.py") == []

    def test_shuffle_false_explicit_produces_no_findings(self, rule):
        source = (
            "import pandas as pd\n"
            "from sklearn.model_selection import train_test_split\n"
            "dates = pd.date_range('2020-01-01', periods=100, freq='B')\n"
            "returns = prices.pct_change()\n"
            "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, test_size=0.3, shuffle=False)\n"
        )
        assert rule.check(source, "correct_split.py") == []

    def test_shuffle_true_always_flagged(self, rule):
        # shuffle=True flagged even without temporal context
        source = (
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = "
            "train_test_split(X, y, shuffle=True)\n"
        )
        findings = rule.check(source, "shuffle_true.py")
        assert len(findings) >= 1
        assert findings[0].rule_id == "RT001"

    def test_no_temporal_context_no_finding(self, rule):
        # No date_range, pct_change, rolling etc. — cross-sectional data
        source = (
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = "
            "train_test_split(X, y, test_size=0.3)\n"
        )
        assert rule.check(source, "cross_sectional.py") == []

    def test_test_file_excluded(self, rule):
        source = (
            "from sklearn.model_selection import train_test_split\n"
            "import pandas as pd\n"
            "dates = pd.date_range('2020-01-01', periods=100, freq='B')\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        )
        assert rule.check(source, "tests/test_model.py") == []

    def test_temporal_context_without_shuffle_flagged(self, rule):
        source = (
            "import pandas as pd\n"
            "from sklearn.model_selection import train_test_split\n"
            "returns = prices.pct_change()\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        )
        findings = rule.check(source, "ts_no_shuffle.py")
        assert len(findings) >= 1
        assert findings[0].rule_id == "RT001"

    def test_rolling_triggers_temporal_context(self, rule):
        source = (
            "from sklearn.model_selection import train_test_split\n"
            "vol = returns.rolling(20).std()\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        )
        findings = rule.check(source, "rolling_context.py")
        assert len(findings) >= 1
