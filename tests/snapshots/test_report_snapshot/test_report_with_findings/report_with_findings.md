# Probus Model Risk Report

**Source:** `backtest.py`

## Overall Score

40/100 — High Risk

## Category Scores

- Conceptual Soundness: 50/100
- Outcomes Analysis: 90/100
- Robustness Testing: 100/100
- Process Verification: 100/100

## Findings

### [CRITICAL] CS001

**File:** `backtest.py` — Line 6

shift(-1) at line 6 reads future observations.

**Recommendation:** Remove the negative shift from signal construction.

**Reference:** SR 11-7 §Conceptual Soundness

### [HIGH] CS002

**File:** `backtest.py` — Line 12

A signal is multiplied with returns without shift(1).

**Recommendation:** Apply signal.shift(1) before multiplying with returns.

**Reference:** SR 11-7 §Conceptual Soundness

### [MEDIUM] OA001

**File:** `backtest.py` — Line 1

No mention of transaction costs found anywhere in this file.

**Recommendation:** Add realistic transaction cost assumptions.

**Reference:** SR 11-7 §Outcomes Analysis

## Limitations

Probus is not a replacement for institutional model validation. It automates selected static checks and does not guarantee absence of methodological issues. Human judgment remains essential.

## References

- Federal Reserve SR 11-7
- BCB Resolução 4557
- López de Prado (2018) — Advances in Financial Machine Learning
- Bailey & López de Prado (2014) — The Probability of Backtest Overfitting