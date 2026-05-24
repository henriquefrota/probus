"""
RT001 — RANDOM_SPLIT_TIME_SERIES

Detects train_test_split() applied to time-series data without explicitly
disabling shuffle, which randomly mixes temporal observations and invalidates
out-of-sample evaluation.

WHY THIS MATTERS
----------------
train_test_split defaults to shuffle=True. In a time-series context, this
means observations from 2023 can appear in the training set while observations
from 2019 appear in the test set. The model effectively trains on the future
and is evaluated on the past.

This is a structural form of look-ahead bias that inflates all out-of-sample
metrics (R², IC, hit rate) because the model has seen contemporaneous or
future information during training. In live trading, where only past data is
available at decision time, the strategy will underperform its backtest.

The correct approach for time series is either:
  - train_test_split(..., shuffle=False)  — preserves temporal order
  - A walk-forward or expanding-window cross-validation scheme (e.g.,
    sklearn's TimeSeriesSplit)

DETECTION LOGIC
---------------
Flags when ALL of the following hold:
  1. A call to train_test_split() is found in the file.
  2. shuffle is NOT set to False:
       - shuffle=True is set (always flag regardless of temporal context)
       - shuffle parameter is absent AND file has temporal context signals
  3. Temporal context signals present (for the absent-shuffle case only):
       - date_range, DatetimeIndex, to_datetime function/class calls
       - pct_change, rolling, resample method calls
       - TimeSeriesSplit usage
       - freq= keyword argument anywhere in the file

Conservative: when shuffle is absent and there is no detectable temporal
context, the rule does not flag — the file may be operating on cross-sectional
data for which random split is appropriate.

KNOWN LIMITATIONS
-----------------
- Cross-function analysis is not performed.
- Temporal context detection is heuristic; unusual column naming may
  produce false negatives.
- Test files are excluded.

REFERENCES
----------
- Federal Reserve SR 11-7, §Robustness Testing (2011)
- BCB Resolução 4557 (2017)
- López de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 7.
"""

import ast
from typing import Optional

from probus.rules.base import Finding, Rule

_TEMPORAL_SIGNALS: frozenset[str] = frozenset({
    "date_range",
    "DatetimeIndex",
    "to_datetime",
    "pct_change",
    "rolling",
    "resample",
    "TimeSeriesSplit",
})


def _is_test_filepath(filepath: str) -> bool:
    normalized = filepath.replace("\\", "/")
    return (
        "/tests/" in normalized
        or "/test_" in normalized
        or normalized.startswith("test_")
    )


def _find_train_test_split_calls(tree: ast.AST) -> list[ast.Call]:
    """Collect all call nodes that invoke train_test_split."""
    result: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name == "train_test_split":
            result.append(node)
    return result


def _get_shuffle_value(call: ast.Call) -> Optional[bool]:
    """
    Return the explicit shuffle keyword value, or None if absent.
      shuffle=True  → True
      shuffle=False → False
      absent        → None
    """
    for kw in call.keywords:
        if kw.arg == "shuffle":
            if isinstance(kw.value, ast.Constant):
                return bool(kw.value.value)
    return None


def _has_temporal_context(tree: ast.AST) -> bool:
    """Return True if the file contains clear time-series usage signals."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _TEMPORAL_SIGNALS:
                return True
            # freq= keyword is a strong temporal indicator
            for kw in node.keywords:
                if kw.arg == "freq":
                    return True
    return False


class RT001RandomSplitTimeSeries(Rule):
    """
    RT001 — RANDOM_SPLIT_TIME_SERIES

    Detects train_test_split() used on time-series data without shuffle=False.

    Bad example::

        # Temporal data — random split destroys time ordering
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    Good example::

        # Explicit shuffle=False preserves temporal order
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=False
        )
    """

    rule_id = "RT001"
    severity = "medium"

    def check(self, source: str, filepath: str) -> list[Finding]:
        if _is_test_filepath(filepath):
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        calls = _find_train_test_split_calls(tree)
        if not calls:
            return []

        temporal = _has_temporal_context(tree)
        findings: list[Finding] = []

        for call in calls:
            shuffle = _get_shuffle_value(call)

            if shuffle is False:
                continue

            if shuffle is True or temporal:
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=self.severity,
                        message=(
                            f"train_test_split at line {call.lineno} "
                            + (
                                "uses shuffle=True, which "
                                if shuffle is True
                                else "does not set shuffle=False. In a file with "
                                "temporal context (date_range, rolling, pct_change, etc.), "
                                "the default shuffle=True "
                            )
                            + "randomly mixes past and future observations. The model "
                            "trains on future data and is evaluated on the past, "
                            "inflating all out-of-sample metrics."
                        ),
                        line=call.lineno,
                        file=filepath,
                        recommendation=(
                            "Pass shuffle=False to train_test_split to preserve "
                            "temporal order: train_test_split(X, y, test_size=0.3, "
                            "shuffle=False). For more rigorous time-series evaluation, "
                            "use sklearn.model_selection.TimeSeriesSplit with a "
                            "walk-forward or expanding-window scheme."
                        ),
                        reference=(
                            "SR 11-7 §Robustness Testing; "
                            "López de Prado (2018) Advances in Financial Machine "
                            "Learning, Ch. 7"
                        ),
                    )
                )

        return findings
