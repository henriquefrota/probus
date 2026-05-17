"""Moving average crossover — correct one-period execution lag.

signal.shift(1) ensures that the position decided at the close of day t
is applied to the return earned on day t+1, reflecting the realistic
constraint that a signal is not actionable until the next trading session.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000
dates = pd.date_range("2016-01-01", periods=n, freq="B")
prices = pd.Series(
    np.random.randn(n).cumsum() * 0.5 + 100,
    index=dates,
    name="close",
)

ma_fast = prices.rolling(10).mean()
ma_slow = prices.rolling(40).mean()
signal = (ma_fast > ma_slow).astype(int)

returns = prices.pct_change()

# CORRECT: shift(1) on the signal provides the one-period execution lag.
# The position determined at the close of t earns the return of t+1.
strategy_returns = signal.shift(1) * returns

cumulative = (1 + strategy_returns).cumprod()
sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
print(f"Sharpe: {sharpe:.2f}")
print(f"Total return: {cumulative.iloc[-1] - 1:.2%}")
