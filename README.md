# Probus

[![CI](https://github.com/henriquefrota/probus/actions/workflows/ci.yml/badge.svg)](https://github.com/henriquefrota/probus/actions/workflows/ci.yml) [![Version](https://img.shields.io/badge/version-0.2.0-orange)](https://github.com/henriquefrota/probus/releases/tag/v0.2.0) [![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Probus is an open-source toolkit that brings selected model risk checks into Python-based quantitative research.**

> Probus is not a replacement for institutional model validation. It automates a small set of static checks and signals what still requires human judgment.

Built by [Henrique Frota](https://github.com/henriquefrota), Economics @ PUC-Rio, as part of ongoing work on model risk practices for quantitative research.

---

## What it is

Look-ahead bias, data leakage, and same-period execution are among the most common methodological errors in quantitative research. They are insidious because they don't crash anything — they silently inflate Sharpe ratios, hit rates, and out-of-sample metrics. A strategy that looks promising in a backtest may be entirely an artifact of one of these errors.

Probus is a static analysis toolkit that codifies a handful of these errors as checks on your research code (`.py` files), so they can be caught at the code-review stage, before they reach a model evaluation report. Each check is AST-based (it parses your code, it does not run it), explains *why* the pattern is a problem, *how* to fix it, and cites the relevant literature.

## What it is not

- **Not a replacement for institutional model validation.** It catches a few well-defined patterns; it does not certify a model.
- **Not a compliance tool.** SR 11-7 and BCB Resolução 4557 inform how the checks are organized — they are not compliance targets.
- **Not a runtime profiler.** Static analysis cannot detect errors that depend on actual data values or only appear at runtime.
- **Not exhaustive.** Six checks today. A clean Probus report means these specific patterns were not found, nothing more.

---

## The four SR 11-7 dimensions

Probus organizes its checks around the four model risk dimensions described in Federal Reserve SR 11-7 and BCB Resolução 4557:

1. **Conceptual Soundness** — *Is the model conceptually correct?*
2. **Outcomes Analysis** — *Are model outputs evaluated against realistic assumptions?*
3. **Robustness Testing** — *Does the model hold under realistic conditions?*
4. **Process Verification** — *Is the process reproducible and auditable?*

Each check maps to exactly one dimension.

---

## The six checks

| ID | Name | Dimension | Severity | What it catches |
|---|---|---|---|---|
| CS001 | `LOOKAHEAD_SHIFT_NEGATIVE` | Conceptual Soundness | critical | A negative `.shift(-N)` that reads future observations into the current row (label/target assignments are excluded). |
| CS002 | `SAME_PERIOD_SIGNAL_EXECUTION` | Conceptual Soundness | high | A signal multiplied by same-period returns with no intervening `shift(1)` — impossible same-bar execution. |
| CS003 | `SCALER_FIT_FULL_DATASET` | Conceptual Soundness | high | An sklearn scaler/transformer fitted on the full dataset before the train/test split, leaking test statistics. |
| OA001 | `NO_TRANSACTION_COSTS` | Outcomes Analysis | medium | A self-contained backtest that reports performance with no mention of transaction costs anywhere in the file. |
| RT001 | `RANDOM_SPLIT_TIME_SERIES` | Robustness Testing | medium | `train_test_split` without `shuffle=False` in a file with time-series context, mixing past and future. |
| PV001 | `MISSING_RANDOM_SEED` | Process Verification | medium | Random number generators used without a fixed seed or `random_state`, breaking reproducibility. |

---

## Installation

Requires Python 3.11+. Probus is not yet published to PyPI — install from source:

```bash
git clone https://github.com/henriquefrota/probus.git
cd probus
pip install -e .
```

Using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

> **Note:** After `pip install -e .`, the `probus` entry-point is placed in your environment's `Scripts/` directory. If that directory is not on your `PATH`, use `python -m probus.cli` as a drop-in replacement for all commands below.

## Usage

```bash
# Audit a single file
python -m probus.cli audit path/to/backtest.py

# Audit a directory (recursive)
python -m probus.cli audit src/models/

# JSON output for CI integration
python -m probus.cli audit path/to/backtest.py --format json

# Print the version
python -m probus.cli --version
```

The command exits with status `1` when any finding is reported, so it can gate a CI pipeline.

---

## Example output

Running Probus on the bundled example (`examples/momentum_backtest.py`) produces a Markdown report:

```
# Probus Model Risk Report

**Source:** `examples/momentum_backtest.py`

## Overall Score

10/100 — Critical Risk

## Category Scores

- Conceptual Soundness: 30/100
- Outcomes Analysis: 90/100
- Robustness Testing: 100/100
- Process Verification: 90/100

## Findings

### [CRITICAL] CS001

**File:** `examples/momentum_backtest.py` — Line 41

shift(-1) at line 41 shifts data backward by 1 period(s), reading future
observations into the current index position. Any signal or feature derived
from this call has look-ahead bias.

**Recommendation:** Remove the negative shift from signal and feature
construction. If you need an execution lag, use shift(1) on the signal side.
If this shift(-N) is creating a supervised learning target (forward return),
assign the result to a variable named 'target', 'y', 'fwd_return', or similar
— those assignments are excluded from this check.

**Reference:** SR 11-7 §Conceptual Soundness; López de Prado (2018) Advances
in Financial Machine Learning, Ch. 7

### [HIGH] CS003

**File:** `examples/momentum_backtest.py` — Line 75

'scaler' is fitted on the full dataset at line 75, before the train/test
split detected at line 78. This leaks test-set statistics (mean, variance)
into the preprocessing step and invalidates the out-of-sample evaluation.

...
```

Each finding carries the rule ID, severity, line, an explanation, a concrete fix, and a reference. The report ends with a limitations note and the reference list. JSON output (`--format json`) carries the same fields for programmatic use.

---

## Limitations

- Probus performs **static analysis**. It cannot detect errors that only manifest at runtime or depend on actual data values.
- Analysis is **file-scoped**. Cross-module and cross-function analysis is not performed; the checks read source-code structure and ordering as a proxy.
- Unusual code structures (for example, a scaler fitted inside a helper called only on training data) may produce false positives. A public benchmark with a measured false-positive rate is planned for v0.3.
- Only `.py` files are analyzed today. **Notebook (`.ipynb`) support is planned for v0.3** — it is not present yet.
- Test files (paths under `tests/` or named `test_*`) are excluded from every check.

---

## Development

```bash
git clone https://github.com/henriquefrota/probus
cd probus
pip install -e ".[dev]"   # or: uv sync --extra dev

pytest                    # run the test suite
ruff check src tests      # lint
black --check src tests   # format check
```

---

## References

- Federal Reserve SR 11-7 — *Guidance on Model Risk Management* (2011)
- BCB Resolução 4557 — *Gerenciamento de Riscos e de Capital* (2017)
- Bailey, D. & López de Prado, M. (2014) — *The Probability of Backtest Overfitting*
- López de Prado, M. (2018) — *Advances in Financial Machine Learning*

---

## License

MIT — see [LICENSE](LICENSE).
