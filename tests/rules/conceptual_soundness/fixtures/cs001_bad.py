"""Momentum signal backtest — look-ahead bias via shift(-1).

A shift(-1) on a feature or signal column reads the NEXT period's value
into the current row. Any model trained or evaluated on this data has
implicitly seen the future, inflating all performance metrics.
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

# Factor construction
momentum_raw = prices.rolling(20).mean() / prices.rolling(60).mean() - 1
volatility = returns.rolling(20).std()

# -----------------------------------------------------------------------
# BUG: shift(-1) on the momentum signal introduces look-ahead bias.
#
# momentum_raw.shift(-1) places the value from row t+1 into row t.
# At decision time t we would not yet have the closing prices needed to
# compute momentum at t+1. The backtest is effectively using tomorrow's
# factor values to make today's trading decision.
# -----------------------------------------------------------------------
signal = momentum_raw.shift(-1)  # look-ahead: uses next-period factor values

# Rank cross-sectionally and go long top decile
ranks = signal.rank(axis=1, pct=True)
positions = (ranks > 0.8).astype(float)
positions = positions.div(positions.sum(axis=1), axis=0).fillna(0)

strategy_returns = (positions.shift(1) * returns).sum(axis=1)

sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
print(f"Annualised Sharpe: {sharpe:.2f}")
