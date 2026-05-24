"""
PV001 — MISSING_RANDOM_SEED

Detects use of random number generators without a fixed seed, which breaks
reproducibility across runs.

WHY THIS MATTERS
----------------
Model validation requires that results be reproducible. A backtest or model
evaluation that uses unseeded random number generators will produce different
outputs each time it is run. This makes it impossible to verify results
independently, track regressions, or compare strategies on equal footing.

SR 11-7 §Process Verification requires that models be documented and
reproducible. An unseeded simulation is a direct violation of this principle:
the reported Sharpe ratio, IC, or R² cannot be confirmed by running the same
code again.

DETECTION LOGIC
---------------
Flags when all of the following are true within the same file:
  1. At least one RNG function call is present:
       - numpy:  np.random.randn, rand, choice, randint, shuffle, permutation
       - stdlib: random.choice, shuffle, sample, random
       - torch:  torch.randn, rand, randint
       - sklearn: train_test_split, RandomForest*, KMeans, and other
                  stochastic estimators
  2. No seed-setting call is found anywhere in the file:
       - np.random.seed(...)
       - np.random.default_rng(N)  (only counts when N is provided)
       - random.seed(...)
       - torch.manual_seed(...)
       - random_state=N as a keyword argument anywhere

The finding is reported at the line of the first detected RNG call.

CONSERVATISM NOTE
-----------------
The rule checks for ANY seed indicator at file scope. A single random_state=42
on any call in the file suppresses the finding, even if other calls lack seeds.
This is intentional: the rule targets files with no seed discipline at all, not
partial seed coverage. Partial coverage is a separate concern.

Numpy's Generator API (np.random.default_rng → rng.randn(...)) is naturally
excluded: RNG calls on a generator instance are not detected as numpy RNG
calls since the receiver is the instance variable, not np.random.

KNOWN LIMITATIONS
-----------------
- Seed set inside a conditional branch (e.g., only in an 'if debug:' block)
  is still counted as a seed — the rule does not model control flow.
- Seeding via environment variable or CLI argument cannot be detected.
- Test files (paths containing 'test_' or '/tests/') are excluded.

REFERENCES
----------
- Federal Reserve SR 11-7, §Process Verification (2011)
- BCB Resolução 4557 (2017)
- reproducibility as a core model risk principle
"""

import ast

from probus.rules.base import Finding, Rule

_NUMPY_RNG_METHODS: frozenset[str] = frozenset({
    "randn", "rand", "choice", "randint", "shuffle", "permutation",
})

_STDLIB_RNG_METHODS: frozenset[str] = frozenset({
    "choice", "shuffle", "sample", "random",
})

_TORCH_RNG_METHODS: frozenset[str] = frozenset({
    "randn", "rand", "randint",
})

_SKLEARN_STOCHASTIC: frozenset[str] = frozenset({
    "train_test_split",
    "RandomForestRegressor", "RandomForestClassifier",
    "GradientBoostingRegressor", "GradientBoostingClassifier",
    "ExtraTreesRegressor", "ExtraTreesClassifier",
    "BaggingRegressor", "BaggingClassifier",
    "AdaBoostRegressor", "AdaBoostClassifier",
    "KMeans", "MiniBatchKMeans",
    "SGDRegressor", "SGDClassifier",
})


def _is_numpy_rng(node: ast.Call) -> bool:
    """np.random.{randn, rand, choice, randint, shuffle, permutation}"""
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in _NUMPY_RNG_METHODS
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "random"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "np"
    )


def _is_stdlib_rng(node: ast.Call) -> bool:
    """random.{choice, shuffle, sample, random}"""
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in _STDLIB_RNG_METHODS
        and isinstance(func.value, ast.Name)
        and func.value.id == "random"
    )


def _is_torch_rng(node: ast.Call) -> bool:
    """torch.{randn, rand, randint}"""
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in _TORCH_RNG_METHODS
        and isinstance(func.value, ast.Name)
        and func.value.id == "torch"
    )


def _is_sklearn_stochastic(node: ast.Call) -> bool:
    """Call to a known stochastic sklearn estimator or splitter."""
    func = node.func
    name = None
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr
    return name in _SKLEARN_STOCHASTIC


def _is_rng_call(node: ast.Call) -> bool:
    return (
        _is_numpy_rng(node)
        or _is_stdlib_rng(node)
        or _is_torch_rng(node)
        or _is_sklearn_stochastic(node)
    )


def _is_numpy_seed(node: ast.Call) -> bool:
    """np.random.seed(...) — any arguments, including none."""
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "seed"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "random"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "np"
    )


def _is_numpy_default_rng_seeded(node: ast.Call) -> bool:
    """np.random.default_rng(N) — only counts when N is explicitly provided."""
    func = node.func
    is_default_rng = (
        isinstance(func, ast.Attribute)
        and func.attr == "default_rng"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "random"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "np"
    )
    return is_default_rng and (bool(node.args) or bool(node.keywords))


def _is_stdlib_seed(node: ast.Call) -> bool:
    """random.seed(...)"""
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "seed"
        and isinstance(func.value, ast.Name)
        and func.value.id == "random"
    )


def _is_torch_seed(node: ast.Call) -> bool:
    """torch.manual_seed(...)"""
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "manual_seed"
        and isinstance(func.value, ast.Name)
        and func.value.id == "torch"
    )


def _is_seed_call(node: ast.Call) -> bool:
    return (
        _is_numpy_seed(node)
        or _is_numpy_default_rng_seeded(node)
        or _is_stdlib_seed(node)
        or _is_torch_seed(node)
        or any(kw.arg == "random_state" for kw in node.keywords)
    )


def _is_test_filepath(filepath: str) -> bool:
    normalized = filepath.replace("\\", "/")
    return (
        "/tests/" in normalized
        or "/test_" in normalized
        or normalized.startswith("test_")
    )


class PV001MissingRandomSeed(Rule):
    """
    PV001 — MISSING_RANDOM_SEED

    Detects files that use random number generators without setting a
    reproducible seed anywhere in the file.

    Bad example::

        import numpy as np
        prices = pd.Series(np.random.randn(1000).cumsum() + 100)  # unseeded

    Good example::

        import numpy as np
        np.random.seed(42)
        prices = pd.Series(np.random.randn(1000).cumsum() + 100)  # reproducible
    """

    rule_id = "PV001"
    severity = "medium"

    def check(self, source: str, filepath: str) -> list[Finding]:
        if _is_test_filepath(filepath):
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        rng_lines: list[int] = []
        seed_found = False

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _is_seed_call(node):
                seed_found = True
            elif _is_rng_call(node):
                rng_lines.append(node.lineno)

        if not rng_lines or seed_found:
            return []

        first_line = min(rng_lines)
        return [
            Finding(
                rule_id=self.rule_id,
                severity=self.severity,
                message=(
                    f"Random number generator functions are used in this file "
                    f"(first occurrence at line {first_line}) but no random seed "
                    "is set anywhere. Without a fixed seed, results are not "
                    "reproducible across runs."
                ),
                line=first_line,
                file=filepath,
                recommendation=(
                    "Add np.random.seed(N), random.seed(N), or torch.manual_seed(N) "
                    "at the top of the file. For scikit-learn estimators, pass "
                    "random_state=N explicitly to all stochastic components "
                    "(train_test_split, RandomForest*, KMeans, etc.)."
                ),
                reference=(
                    "SR 11-7 §Process Verification; "
                    "reproducibility as a core model risk principle"
                ),
            )
        ]
