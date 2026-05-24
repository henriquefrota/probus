# Probus

[![Tests](https://img.shields.io/badge/tests-112%20passing-brightgreen)](https://github.com/henriquefrota/probus/tree/main/tests) [![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/henriquefrota/probus/blob/main/LICENSE) [![Release](https://img.shields.io/badge/release-v0.2.0-orange)](https://github.com/henriquefrota/probus/releases/tag/v0.2.0)

**Probus is an open-source toolkit for model risk checks in Python-based quantitative research.**

> Probus is not a replacement for institutional model validation. It is an open-source toolkit that brings selected model risk checks into Python-based quantitative research workflows.

Built by [Henrique Frota](https://github.com/henriquefrota), Economics @ PUC-Rio. Part of ongoing work on model risk practices for quantitative research.

---

## Why this matters

Look-ahead bias, data leakage, and same-period execution are among the most common methodological errors in quantitative research. They are especially insidious because they don't cause crashes or visible bugs — they silently inflate Sharpe ratios, hit rates, and out-of-sample metrics. A strategy that looks promising in a backtest may be entirely an artifact of one of these errors.

Probus codifies the most common of these errors as static analysis rules so they can be caught at the code-review stage, before they reach a model evaluation report.

---

## Quick example

Consider this snippet — a momentum strategy that looks profitable in backtest:

```python
import pandas as pd

prices = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")

# Momentum signal
signal = prices["close"].rolling(20).mean().shift(-1)   # ← bug

# Compute strategy returns
returns = prices["close"].pct_change()
strategy = signal * returns

print(f"Sharpe: {strategy.mean() / strategy.std() * 252**0.5:.2f}")
```

Running Probus on this file:

```
$ python -m probus.cli audit backtest.py

# Probus Model Risk Report

**Source:** `backtest.py`

## Overall Score
70/100 — Moderate Risk

## Findings

### [CRITICAL] CS001
**File:** `backtest.py` — Line 6

shift(-1) at line 6 shifts data backward by 1 period(s), reading future
observations into the current index position. Any signal or feature
derived from this call has look-ahead bias.

**Recommendation:** Remove the negative shift from signal and feature
construction. If you need an execution lag, use shift(1) on the signal
side. If this shift(-N) is creating a supervised learning target
(forward return), assign the result to a variable named 'target', 'y',
'fwd_return', or similar — those assignments are excluded from this check.

**Reference:** SR 11-7 §Conceptual Soundness; López de Prado (2018)
Advances in Financial Machine Learning, Ch. 7
```

The `shift(-1)` on the signal pulls tomorrow's moving average into today's row — the strategy is using future information to make present decisions. Without Probus, this bug typically surfaces only when the strategy fails in live trading.

---

## What it does

Probus organizes its rules around the four model risk dimensions described in Federal Reserve SR 11-7 and BCB Resolução 4557: Conceptual Soundness, Outcomes Analysis, Robustness Testing, and Process Verification. Each rule maps to one of these dimensions and includes its rationale, fix, and academic or regulatory reference.

**Probus is:**
- A static analysis toolkit for quantitative research code (`.py` files)
- Educational by design — each rule explains *why* the pattern is problematic, *how* to fix it, and cites the relevant literature
- Open-source, auditable, and MIT-licensed

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

> **Note:** After `pip install -e .`, the `probus` entry-point script is placed in your Python environment's `Scripts/` directory. If that directory is not on your `PATH`, use `python -m probus.cli` as a drop-in replacement for all commands below.

---

## Usage

```bash
# Audit a single file
python -m probus.cli audit path/to/backtest.py

# Audit a directory (recursive)
python -m probus.cli audit src/models/

# JSON output for CI integration
python -m probus.cli audit path/to/backtest.py --format json
```

---

## Rules (v0.2)

Rules are organized by the four model risk dimensions from SR 11-7.

### Conceptual Soundness — *Is the model conceptually correct?*

| ID | Name | Severity | Description |
|---|---|---|---|
| CS001 | `LOOKAHEAD_SHIFT_NEGATIVE` | critical | Detects `.shift(N)` calls with a negative integer argument, which read future observations into the current row. Any signal or feature derived from such a call has look-ahead bias and will produce inflated backtest results that cannot be replicated in live trading. |
| CS002 | `SAME_PERIOD_SIGNAL_EXECUTION` | high | Detects signals combined arithmetically with returns of the same period, with no intervening `shift(1)`. This implies the signal was both generated and executed within the same bar — impossible in practice, since a signal computed from a bar's close cannot be acted upon until the next bar opens. |
| CS003 | `SCALER_FIT_FULL_DATASET` | high | Detects sklearn-compatible scalers or transformers fitted on the full dataset before the train/test split is performed. This leaks statistics (mean, variance, quantiles) from the test partition into the preprocessing step, invalidating any out-of-sample evaluation. |

### Outcomes Analysis — *Are model outputs evaluated against realistic assumptions?*

| ID | Name | Severity | Description |
|---|---|---|---|
| OA001 | `NO_TRANSACTION_COSTS` | medium | Detects complete backtests that calculate strategy returns and evaluate performance with no mention of transaction costs anywhere in the file. Inflated Sharpe ratios and cumulative returns are a typical consequence. (File-level check.) |

### Robustness Testing — *Does the model hold under realistic conditions?*

| ID | Name | Severity | Description |
|---|---|---|---|
| RT001 | `RANDOM_SPLIT_TIME_SERIES` | medium | Detects `train_test_split` called with `shuffle=True` (or without `shuffle=False`) in files with clear temporal/time-series context. Random splitting of time-ordered data leaks future observations into the training set. |

### Process Verification — *Is the process reproducible and auditable?*

| ID | Name | Severity | Description |
|---|---|---|---|
| PV001 | `MISSING_RANDOM_SEED` | medium | Detects use of random number generators (numpy, stdlib random, torch, sklearn) without a fixed seed or `random_state`, breaking reproducibility of results. |

---

## Roadmap

- **v0.1** ✓ complete — Conceptual Soundness rules: CS001, CS002, CS003
- **v0.2** ✓ complete — Outcomes Analysis (OA001), Robustness Testing (RT001), Process Verification (PV001)
- **v0.3** — Notebook (.ipynb) support, public benchmark with measured false positive rate, MkDocs documentation site
- **v1.0** — Consolidated rule set (~8 rules), bilingual documentation (EN/PT)
- **Post-v1.0** — Experimental statistics module (Deflated Sharpe Ratio, Probability of Backtest Overfitting, CSCV) as opt-in extension

Roadmap is indicative, not contractual. Priorities shift based on discovery conversations with practitioners.

---

## Project philosophy

**Rigor.** Every rule has a technical justification and a published reference. Nothing is flagged on intuition alone.

**Humility.** Probus automates what static analysis can automate and signals what requires human judgment. Limitations are part of the documentation, not a footnote.

**Transparency.** Rules are auditable. Scores are computed with documented formulas. There are no black-box assessments.

---

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/henriquefrota/probus
cd probus
uv sync --extra dev

# Run the test suite
pytest

# Run a local audit
python -m probus.cli audit path/to/file.py
```

---

## Limitations

- Probus performs **static analysis**. It cannot detect errors that only manifest at runtime or depend on actual data values.
- Analysis is **file-scoped**. Cross-module and cross-notebook analysis is not yet supported.
- The false positive target is **< 5% on clean code**. Unusual code structures (e.g., scaler fit inside a helper function called only on training data) may produce false positives. See individual rule documentation for known edge cases.
- Probus is **not a compliance tool**. SR 11-7 and BCB Resolução 4557 are referenced as inspiration for the conceptual structure, not as compliance targets.

---

## References

- Federal Reserve SR 11-7 — *Guidance on Model Risk Management* (2011)
- BCB Resolução 4557 — *Gerenciamento de Riscos e de Capital* (2017)
- López de Prado, M. (2018) — *Advances in Financial Machine Learning*
- Bailey, D. & López de Prado, M. (2014) — *The Probability of Backtest Overfitting*

---

## License

MIT — see [LICENSE](LICENSE).
