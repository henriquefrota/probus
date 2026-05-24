# Probus Demo: Catching Bugs in a Momentum Backtest

This example shows Probus v0.1 detecting three methodological errors in a
realistic cross-sectional momentum strategy. The code looks plausible at first
glance — it follows standard pandas idioms, uses scikit-learn for an ML
overlay, and prints a Sharpe ratio. All three bugs are silent: the code runs
without errors and produces a number.

---

## The bugs (high-level)

1. **Look-ahead bias (CS001):** A feature used to construct the momentum
   signal incorporates a `.shift(-1)` that reads tomorrow's value into
   today's row. The inflated signal makes the strategy appear more predictive
   than it is.

2. **Same-period signal execution (CS002):** Strategy returns are computed by
   multiplying the signal directly with same-period returns, without the
   one-period execution lag that live trading requires.

3. **Scaler leakage (CS003):** A `StandardScaler` is fitted on the full
   feature matrix before the train/test split. Test-set statistics (mean,
   variance) leak into the preprocessing step, invalidating the ML overlay's
   out-of-sample evaluation.

---

## Running Probus

```bash
python -m probus.cli audit examples/momentum_backtest.py
```

Expected output:

```
# Probus Model Risk Report

**Source:** `examples/momentum_backtest.py`

## Overall Score

30/100 — Critical Risk

## Findings

### [CRITICAL] CS001
**File:** `examples/momentum_backtest.py` — Line 43

shift(-1) at line 43 shifts data backward by 1 period(s), reading future
observations into the current index position. Any signal or feature derived
from this call has look-ahead bias.

### [HIGH] CS002
**File:** `examples/momentum_backtest.py` — Line 88

At line 88, a signal is multiplied with period returns without an intervening
shift(1). This implies the signal was generated and acted upon within the same
bar, which is impossible in live trading.

### [HIGH] CS003
**File:** `examples/momentum_backtest.py` — Line 77

'scaler' is fitted on the full dataset at line 77, before the train/test split
detected at line 80. This leaks test-set statistics (mean, variance) into the
preprocessing step and invalidates the out-of-sample evaluation.
```

---

## The legitimate use: `fwd_return = returns.shift(-1)`

Line 46 also uses `.shift(-1)`, but on a variable named `fwd_return` — a
supervised learning target. Probus correctly does **not** flag this: CS001
excludes `shift(-N)` calls assigned to label-like variable names (`fwd_return`,
`target`, `y`, etc.), because creating a forward-return target is the intended
use of a negative shift.

---

## Why this matters

A researcher running this backtest would see an inflated Sharpe ratio and
proceed to model evaluation. The bugs do not cause exceptions or warnings —
they silently improve the apparent results. A strategy built on this code would
pass internal review, fail in live trading, and the failure would be difficult
to trace back to the methodology without a tool like Probus.
