"""All rules share one test-file exclusion policy (see rules/_utils.py).

Production-code rules must not flag intentionally broken code that lives in
test files or under a tests/ tree. This pins that uniform behavior across all
six rules so the policy cannot silently diverge again.
"""

import pytest

from probus.rules import get_all_rules
from probus.rules._utils import is_test_filepath

# Source that, in a production file, trips at least one rule each.
_TRIGGERING_SOURCE = (
    "import numpy as np\n"
    "from sklearn.model_selection import train_test_split\n"
    "signal = df['close'].rolling(20).mean().shift(-1)\n"
    "returns = df['close'].pct_change()\n"
    "strategy_returns = signal * returns\n"
    "X_train, X_test = train_test_split(signal, shuffle=True)\n"
    "noise = np.random.randn(100)\n"
    "sharpe = strategy_returns.mean() / strategy_returns.std() * 252\n"
)

_TEST_PATHS = [
    "tests/test_backtest.py",
    "test_backtest.py",
    "a/b/tests/fixtures/cs001_bad.py",
    "src/test_helper.py",
]


def test_production_path_is_flagged():
    findings = [
        f
        for rule in get_all_rules()
        for f in rule.check(_TRIGGERING_SOURCE, "src/models/backtest.py")
    ]
    assert findings, "Expected the triggering source to be flagged in a production file"


@pytest.mark.parametrize("path", _TEST_PATHS)
def test_test_paths_are_skipped_by_every_rule(path):
    assert is_test_filepath(path)
    for rule in get_all_rules():
        assert rule.check(_TRIGGERING_SOURCE, path) == [], (
            f"{rule.rule_id} should skip test path {path!r}"
        )
