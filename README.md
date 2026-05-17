# Probus

**Probus is an open-source toolkit for model risk checks in Python-based quantitative research.**

> Probus is not a replacement for institutional model validation. It is an open-source toolkit that brings selected model risk checks into Python-based quantitative research workflows.

---

## What it does

Backtests and quantitative models routinely suffer from methodological errors that inflate results and generate false confidence: look-ahead bias, data leakage, missing transaction costs, and weak validation patterns. Probus automates static checks for a curated set of these errors, organized around the model risk dimensions described in Federal Reserve SR 11-7 and BCB Resolução 4557.

**Probus is not:**
- A trading tool or signal generator
- A complete model governance framework
- A compliance tool for SR 11-7 or BCB 4557
- A replacement for institutional model validation departments

**Probus is:**
- A static analysis toolkit for quantitative research code (`.py` files)
- Educational by design — each rule explains *why* the pattern is problematic, *how* to fix it, and cites the relevant literature
- Open-source, auditable, and MIT-licensed

---

## Installation

Requires Python 3.11+.

```bash
pip install probus
```

Using [uv](https://github.com/astral-sh/uv):

```bash
uv add probus
```

---

## Usage

```bash
# Audit a single file
probus audit path/to/backtest.py

# Audit a directory (recursive)
probus audit src/models/

# JSON output for CI integration
probus audit path/to/backtest.py --format json
```

---

## Rules (v0.1)

Rules are organized by the four model risk dimensions from SR 11-7.

### Conceptual Soundness — *Is the model conceptually correct?*

| ID | Name | Severity | Description |
|---|---|---|---|
| CS003 | `SCALER_FIT_FULL_DATASET` | high | A sklearn-compatible scaler or transformer is fitted on the full dataset before the train/test split, leaking test-set statistics into the preprocessing step. |

Rules CS001 (`LOOKAHEAD_SHIFT_NEGATIVE`) and CS002 (`SAME_PERIOD_SIGNAL_EXECUTION`) are planned for v0.2.

### Outcomes Analysis, Robustness Testing, Process Verification

Rules for these dimensions (`OA001`, `RT001`, `PV001`) are planned for v0.2.

---

## Report structure

```
# Probus Model Risk Report

## Overall Score
80/100 — Low Risk

## Category Scores
- Conceptual Soundness: 80/100
- Outcomes Analysis: 100/100
- Robustness Testing: 100/100
- Process Verification: 100/100

## Findings

### [HIGH] CS003
File: `backtest.py` — Line 47
'scaler' is fitted on the full dataset at line 47, before the train/test
split detected at line 52. ...

## Limitations
Probus is not a replacement for institutional model validation. ...
```

---

## Project philosophy

**Rigor.** Every rule has a technical justification and a published reference. Nothing is flagged on intuition alone.

**Humility.** Probus automates what static analysis can automate and signals what requires human judgment. Limitations are part of the documentation, not a footnote.

**Transparency.** Rules are auditable. Scores are computed with documented formulas. There are no black-box assessments.

---

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/henriqfrota/probus
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
