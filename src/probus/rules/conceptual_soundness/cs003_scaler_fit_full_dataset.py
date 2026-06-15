"""
CS003 — SCALER_FIT_FULL_DATASET

Detects sklearn-compatible scalers or transformers that are fitted on the
full dataset before a temporal or holdout split is performed.

WHY THIS MATTERS
----------------
When a scaler (e.g., StandardScaler) is fitted on the full dataset and the
data is split afterwards, the mean and standard deviation used to normalize
the test set are computed from test-set observations. This is a form of
data leakage: the model's preprocessing has implicitly "seen the future,"
violating the core principle that no information from the validation period
should influence the model's construction or calibration.

In quantitative finance, this error inflates in-sample performance metrics
and produces overly optimistic out-of-sample estimates. Strategies that
look profitable in backtests built this way routinely fail in live trading.

DETECTION LOGIC
---------------
Flags when, within the same file:
  1. A known scaler/transformer variable is instantiated.
  2. `.fit()` or `.fit_transform()` is called on that variable.
  3. The fit call appears at a line number BEFORE a recognized split pattern.
  4. A split pattern exists later in the file.

If no split pattern is detected at all, the rule does not fire — the scaler
may be intentionally fitted on all available data (e.g., final deployment
after offline validation is complete).

Split patterns recognized:
  - train_test_split(...) or TimeSeriesSplit(...) calls
  - Tuple assignments whose targets include X_train, X_test, y_train,
    y_test, train_data, test_data, train_idx, test_idx, train_X, test_X,
    train_y, or test_y

KNOWN LIMITATIONS
-----------------
- Cross-function analysis is not performed. If the scaler fit and the split
  occur in different functions, the rule may misinterpret the execution order.
  This can produce false positives (fit inside a helper function that is
  called only on training data) or false negatives (split in a wrapper not
  yet recognized). This limitation is documented here rather than guarded
  against with heuristics that would increase false positive rate.
- Pipeline([("scaler", StandardScaler()), ...]) is not yet detected.
- The rule checks source-code line ordering as a proxy for execution order.
  This approximation is valid for linear scripts and single-function bodies,
  which cover the majority of backtest code in practice.
- Test files are excluded.

REFERENCES
----------
- Federal Reserve SR 11-7, §Conceptual Soundness (2011)
- BCB Resolução 4557 (2017)
- López de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 7.
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and
  TensorFlow, Ch. 2.
"""

import ast
from typing import Optional

from probus.rules._utils import is_test_filepath
from probus.rules.base import Finding, Rule

# Sklearn-compatible transformers that must only be fitted on training data
# in a supervised learning or backtesting workflow.
_KNOWN_SCALERS: frozenset[str] = frozenset(
    {
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler",
        "MaxAbsScaler",
        "Normalizer",
        "PowerTransformer",
        "QuantileTransformer",
        "PCA",
    }
)

# Methods that both fit parameters to data and (optionally) transform it.
# Calling these before splitting leaks test-set statistics.
_FIT_METHODS: frozenset[str] = frozenset({"fit", "fit_transform"})

# Sklearn utilities that perform or imply a train/test split.
_SPLIT_FUNCTION_NAMES: frozenset[str] = frozenset(
    {
        "train_test_split",
        "TimeSeriesSplit",
    }
)

# Variable names that conventionally hold a training or test partition.
# Presence of any of these in an assignment is treated as a split signal.
_SPLIT_VARIABLE_NAMES: frozenset[str] = frozenset(
    {
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "train_data",
        "test_data",
        "train_idx",
        "test_idx",
        "train_X",
        "test_X",
        "train_y",
        "test_y",
    }
)


def _callable_name(node: ast.Call) -> Optional[str]:
    """Return the bare name of a callable, or None if it cannot be determined."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _collect_scaler_vars(tree: ast.AST) -> dict[str, int]:
    """
    Walk the AST and return {variable_name: line_number} for every assignment
    that binds a known scaler/transformer to a name.

    Covers both plain and annotated assignments:
      scaler = StandardScaler()
      scaler: StandardScaler = StandardScaler()
    """
    result: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if _callable_name(node.value) in _KNOWN_SCALERS:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        result[target.id] = node.lineno
        elif (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and isinstance(node.value, ast.Call)
        ):
            if _callable_name(node.value) in _KNOWN_SCALERS and isinstance(
                node.target, ast.Name
            ):
                result[node.target.id] = node.lineno
    return result


def _collect_fit_calls(
    tree: ast.AST,
    scaler_vars: dict[str, int],
) -> list[tuple[str, int]]:
    """
    Return (identifier, line_number) for every .fit() / .fit_transform() call
    whose receiver is a tracked scaler variable or an anonymous scaler instantiation.

    Two forms are detected:
      scaler.fit_transform(X)          — named variable
      StandardScaler().fit_transform(X) — anonymous (inline) instantiation
    """
    result: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in _FIT_METHODS):
            continue
        obj = func.value
        if isinstance(obj, ast.Name) and obj.id in scaler_vars:
            # Named variable: scaler.fit_transform(X)
            result.append((obj.id, node.lineno))
        elif isinstance(obj, ast.Call) and _callable_name(obj) in _KNOWN_SCALERS:
            # Anonymous: StandardScaler().fit_transform(X)
            result.append((_callable_name(obj), node.lineno))  # type: ignore[arg-type]
    return result


def _collect_split_lines(tree: ast.AST) -> list[int]:
    """
    Return line numbers where a recognized train/test split pattern appears.

    Detected patterns:
      - Calls to train_test_split(...) or TimeSeriesSplit(...)
      - Assignments (plain or tuple-unpacking) that bind a known split
        variable name such as X_train, X_test, y_train, y_test, etc.
    """
    result: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if _callable_name(node) in _SPLIT_FUNCTION_NAMES:
                result.append(node.lineno)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _SPLIT_VARIABLE_NAMES:
                    result.append(node.lineno)
                    break
                if isinstance(target, ast.Tuple) and any(
                    isinstance(elt, ast.Name) and elt.id in _SPLIT_VARIABLE_NAMES
                    for elt in target.elts
                ):
                    result.append(node.lineno)
                    break
    return result


class CS003ScalerFitFullDataset(Rule):
    """
    CS003 — SCALER_FIT_FULL_DATASET

    Detects when a sklearn-compatible scaler or transformer is fitted on the
    full dataset before the train/test or temporal validation split.

    Bad example::

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)    # BUG: fitted on full data
        X_train, X_test = X_scaled[:n], X_scaled[n:]

    Good example::

        X_train, X_test = X[:n], X[n:]        # split first
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
    """

    rule_id = "CS003"
    severity = "high"

    def check(self, source: str, filepath: str) -> list[Finding]:
        if is_test_filepath(filepath):
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        scaler_vars = _collect_scaler_vars(tree)
        if not scaler_vars:
            return []

        fit_calls = _collect_fit_calls(tree, scaler_vars)
        if not fit_calls:
            return []

        split_lines = _collect_split_lines(tree)
        if not split_lines:
            # No split detected anywhere in the file. The scaler may be
            # intentionally fitted on all available data (e.g., production
            # deployment). Do not flag — conservatism over recall.
            return []

        earliest_split = min(split_lines)
        findings: list[Finding] = []

        for identifier, fit_line in fit_calls:
            if fit_line < earliest_split:
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=self.severity,
                        message=(
                            f"'{identifier}' is fitted on the full dataset at line "
                            f"{fit_line}, before the train/test split detected at line "
                            f"{earliest_split}. This leaks test-set statistics (mean, "
                            "variance) into the preprocessing step and invalidates the "
                            "out-of-sample evaluation."
                        ),
                        line=fit_line,
                        file=filepath,
                        recommendation=(
                            "Split your data first, then call scaler.fit_transform() only "
                            "on the training partition. Apply scaler.transform() — not "
                            "fit_transform() — to the test partition so that scaling "
                            "parameters are derived exclusively from training observations."
                        ),
                        reference=(
                            "SR 11-7 §Conceptual Soundness; "
                            "López de Prado (2018) Advances in Financial Machine Learning, Ch. 7"
                        ),
                    )
                )

        return findings
