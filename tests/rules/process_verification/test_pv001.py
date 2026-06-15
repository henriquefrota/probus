from pathlib import Path

import pytest

from probus.rules.base import Finding
from probus.rules.process_verification.pv001_missing_random_seed import (
    PV001MissingRandomSeed,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def rule() -> PV001MissingRandomSeed:
    return PV001MissingRandomSeed()


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestPV001BadFixture:
    def test_flags_at_least_one_finding(self, rule):
        findings = rule.check(_load("pv001_bad.py"), "pv001_bad.py")
        assert len(findings) >= 1, "Expected PV001 to flag the bad fixture"

    def test_finding_rule_id(self, rule):
        findings = rule.check(_load("pv001_bad.py"), "pv001_bad.py")
        assert findings[0].rule_id == "PV001"

    def test_finding_severity_is_medium(self, rule):
        findings = rule.check(_load("pv001_bad.py"), "pv001_bad.py")
        assert findings[0].severity == "medium"

    def test_finding_line_is_positive(self, rule):
        findings = rule.check(_load("pv001_bad.py"), "pv001_bad.py")
        assert isinstance(findings[0].line, int) and findings[0].line > 0

    def test_finding_line_contains_rng_call(self, rule):
        findings = rule.check(_load("pv001_bad.py"), "pv001_bad.py")
        source_lines = _load("pv001_bad.py").splitlines()
        flagged = source_lines[findings[0].line - 1]
        assert any(
            m in flagged for m in ("randn", "rand(", "randint", "choice", "shuffle")
        ), f"Line {findings[0].line} does not contain a recognizable RNG call: {flagged!r}"

    def test_finding_message_mentions_seed(self, rule):
        findings = rule.check(_load("pv001_bad.py"), "pv001_bad.py")
        assert "seed" in findings[0].message.lower()

    def test_finding_has_recommendation(self, rule):
        findings = rule.check(_load("pv001_bad.py"), "pv001_bad.py")
        assert len(findings[0].recommendation) > 0

    def test_finding_has_reference(self, rule):
        findings = rule.check(_load("pv001_bad.py"), "pv001_bad.py")
        assert len(findings[0].reference) > 0

    def test_finding_file_matches_input(self, rule):
        findings = rule.check(_load("pv001_bad.py"), "pv001_bad.py")
        assert findings[0].file == "pv001_bad.py"

    def test_finding_is_dataclass(self, rule):
        findings = rule.check(_load("pv001_bad.py"), "pv001_bad.py")
        assert isinstance(findings[0], Finding)


class TestPV001GoodFixture:
    def test_produces_zero_findings(self, rule):
        findings = rule.check(_load("pv001_good.py"), "pv001_good.py")
        assert len(findings) == 0, (
            f"Expected zero PV001 findings on good fixture, got {len(findings)}:\n"
            + "\n".join(f"  line {f.line}: {f.message}" for f in findings)
        )


class TestPV001EdgeCases:
    def test_returns_empty_on_syntax_error(self, rule):
        assert rule.check("def (bad syntax", "x.py") == []

    def test_no_rng_calls_produces_no_findings(self, rule):
        source = "import numpy as np\nimport pandas as pd\n"
        assert rule.check(source, "clean.py") == []

    def test_numpy_seed_suppresses_finding(self, rule):
        source = (
            "import numpy as np\n" "np.random.seed(42)\n" "x = np.random.randn(100)\n"
        )
        assert rule.check(source, "seeded.py") == []

    def test_random_state_kwarg_suppresses_finding(self, rule):
        source = (
            "import numpy as np\n"
            "from sklearn.ensemble import RandomForestRegressor\n"
            "x = np.random.randn(100)\n"
            "model = RandomForestRegressor(random_state=42)\n"
        )
        assert rule.check(source, "with_random_state.py") == []

    def test_test_file_path_excluded(self, rule):
        source = "import numpy as np\nx = np.random.randn(100)\n"
        assert rule.check(source, "tests/test_strategy.py") == []

    def test_test_underscore_prefix_excluded(self, rule):
        source = "import numpy as np\nx = np.random.randn(100)\n"
        assert rule.check(source, "test_backtest.py") == []

    def test_default_rng_with_seed_suppresses_finding(self, rule):
        source = (
            "import numpy as np\n"
            "rng = np.random.default_rng(42)\n"
            "x = np.random.randn(100)\n"
        )
        assert rule.check(source, "default_rng.py") == []

    def test_default_rng_without_seed_does_not_suppress(self, rule):
        source = (
            "import numpy as np\n"
            "rng = np.random.default_rng()\n"
            "x = np.random.randn(100)\n"
        )
        findings = rule.check(source, "unseeded_rng.py")
        assert len(findings) >= 1

    def test_stdlib_random_without_seed_flagged(self, rule):
        source = "import random\n" "choices = random.choice([1, 2, 3])\n"
        findings = rule.check(source, "stdlib_rng.py")
        assert len(findings) >= 1
        assert findings[0].rule_id == "PV001"
