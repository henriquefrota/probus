"""Moving average crossover — performance metrics without friction adjustments.

The strategy computes a signal, applies it to returns, and reports an
annualised Sharpe ratio. No adjustments for trading friction are modelled.
The reported Sharpe is the gross theoretical maximum.
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

strategy_returns = signal.shift(1) * returns

cumulative = (1 + strategy_returns).cumprod()
ann_ret = strategy_returns.mean() * 252
ann_vol = strategy_returns.std() * np.sqrt(252)
sharpe = ann_ret / ann_vol

print(f"Sharpe: {sharpe:.2f}")
print(f"Final NAV: {cumulative.iloc[-1]:.4f}")
