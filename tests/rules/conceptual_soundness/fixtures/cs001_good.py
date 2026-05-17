"""Momentum signal backtest — correct implementation.

The signal uses only data available at the decision time (no shift(-1)).
shift(-1) appears only once, to create a supervised learning target (the
forward return we want to predict), which is a legitimate use and is
excluded from CS001.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 500
dates = pd.date_range("2018-01-01", periods=n, freq="B")

prices = pd.DataFrame(
    np.random.randn(n, 10).cumsum(axis=0) + 100,
    index=dates,
    columns=[f"asset_{i}" for i in range(10)],
)

returns = prices.pct_change().dropna()

# Factor construction — no negative shifts on any feature
momentum_raw = prices.rolling(20).mean() / prices.rolling(60).mean() - 1
volatility = returns.rolling(20).std()

# CORRECT: Signal uses only current and past data.
signal = momentum_raw / (volatility + 1e-9)

# CORRECT: Rank-based position construction using current-period signal.
ranks = signal.rank(axis=1, pct=True)
positions = (ranks > 0.8).astype(float)
positions = positions.div(positions.sum(axis=1), axis=0).fillna(0)

# CORRECT: shift(1) on positions to simulate one-period execution lag.
strategy_returns = (positions.shift(1) * returns).sum(axis=1)

sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
print(f"Annualised Sharpe: {sharpe:.2f}")

# CORRECT: shift(-1) is used only to create a supervised learning target.
# This is an intentional forward-looking label, not a signal feature.
# CS001 excludes assignments to variables named 'fwd_return', 'target', etc.
fwd_return = returns.mean(axis=1).shift(-1)
