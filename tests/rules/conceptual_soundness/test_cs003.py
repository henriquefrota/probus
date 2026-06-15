from pathlib import Path

import pytest

from probus.rules.base import Finding
from probus.rules.conceptual_soundness.cs003_scaler_fit_full_dataset import (
    CS003ScalerFitFullDataset,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def rule() -> CS003ScalerFitFullDataset:
    return CS003ScalerFitFullDataset()


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestCS003BadFixture:
    def test_flags_at_least_one_finding(self, rule):
        findings = rule.check(_load("cs003_bad.py"), "cs003_bad.py")
        assert (
            len(findings) >= 1
        ), "Expected CS003 to flag the bad fixture; got 0 findings"

    def test_finding_rule_id(self, rule):
        findings = rule.check(_load("cs003_bad.py"), "cs003_bad.py")
        assert findings[0].rule_id == "CS003"

    def test_finding_severity(self, rule):
        findings = rule.check(_load("cs003_bad.py"), "cs003_bad.py")
        assert findings[0].severity == "high"

    def test_finding_line_is_positive(self, rule):
        findings = rule.check(_load("cs003_bad.py"), "cs003_bad.py")
        assert isinstance(findings[0].line, int)
        assert findings[0].line > 0

    def test_finding_points_to_fit_call(self, rule):
        """The reported line must be the fit call, not the split."""
        findings = rule.check(_load("cs003_bad.py"), "cs003_bad.py")
        source_lines = _load("cs003_bad.py").splitlines()
        flagged_line = source_lines[findings[0].line - 1]
        assert (
            "fit" in flagged_line
        ), f"Line {findings[0].line} does not contain a fit call: {flagged_line!r}"

    def test_finding_message_is_informative(self, rule):
        findings = rule.check(_load("cs003_bad.py"), "cs003_bad.py")
        msg = findings[0].message.lower()
        assert "scaler" in msg or "fit" in msg

    def test_finding_has_recommendation(self, rule):
        findings = rule.check(_load("cs003_bad.py"), "cs003_bad.py")
        assert isinstance(findings[0].recommendation, str)
        assert len(findings[0].recommendation) > 0

    def test_finding_has_reference(self, rule):
        findings = rule.check(_load("cs003_bad.py"), "cs003_bad.py")
        assert isinstance(findings[0].reference, str)
        assert len(findings[0].reference) > 0

    def test_finding_file_matches_input(self, rule):
        findings = rule.check(_load("cs003_bad.py"), "cs003_bad.py")
        assert findings[0].file == "cs003_bad.py"


class TestCS003GoodFixture:
    def test_produces_zero_findings(self, rule):
        findings = rule.check(_load("cs003_good.py"), "cs003_good.py")
        assert len(findings) == 0, (
            f"Expected zero CS003 findings on the good fixture, got {len(findings)}:\n"
            + "\n".join(f"  line {f.line}: {f.message}" for f in findings)
        )


class TestCS003EdgeCases:
    def test_returns_empty_on_syntax_error(self, rule):
        findings = rule.check("def (broken syntax", "bad.py")
        assert findings == []

    def test_returns_empty_when_no_scaler_present(self, rule):
        source = "import numpy as np\nX_train, X_test = X[:n], X[n:]\n"
        findings = rule.check(source, "no_scaler.py")
        assert findings == []

    def test_returns_empty_when_no_split_present(self, rule):
        # Scaler fitted with no split — intentional (e.g., production deployment).
        source = (
            "from sklearn.preprocessing import StandardScaler\n"
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X)\n"
        )
        findings = rule.check(source, "no_split.py")
        assert findings == []

    def test_finding_is_dataclass(self, rule):
        findings = rule.check(_load("cs003_bad.py"), "cs003_bad.py")
        assert isinstance(findings[0], Finding)
