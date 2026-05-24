"""Moving average crossover — Sharpe reported net of transaction costs.

A round-trip commission of 10 bps is applied to each signal change.
The reported Sharpe is net of these trading costs.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000
dates = pd.date_range("2018-01-01", periods=n, freq="B")
prices = pd.Series(np.random.randn(n).cumsum() + 100, index=dates)
returns = prices.pct_change()

ma_fast = prices.rolling(10).mean()
ma_slow = prices.rolling(40).mean()
signal = (ma_fast > ma_slow).astype(int)

commission = 0.001  # 10 bps round-trip
turnover = signal.diff().abs()

gross_returns = signal.shift(1) * returns
strategy_returns = gross_returns - commission * turnover

cumulative = (1 + strategy_returns).cumprod()
ann_ret = strategy_returns.mean() * 252
ann_vol = strategy_returns.std() * np.sqrt(252)
sharpe = ann_ret / ann_vol

print(f"Sharpe (net): {sharpe:.2f}")
print(f"Final NAV:    {cumulative.iloc[-1]:.4f}")
