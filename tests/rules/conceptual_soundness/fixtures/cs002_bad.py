"""Moving average crossover — same-period signal execution.

The strategy multiplies the signal at time t directly with the return at
time t. Since day t's return is already determined before the signal can
be computed and acted upon, this overstates performance by using a return
that the strategy could not realistically have captured.
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

# Moving average signals
ma_fast = prices.rolling(10).mean()
ma_slow = prices.rolling(40).mean()
signal = (ma_fast > ma_slow).astype(int)

# Daily log returns
returns = prices.pct_change()

# -----------------------------------------------------------------------
# BUG: signal and returns are from the same period.
#
# signal[t] is computed using closing prices at t. The return at t measures
# the price move from t-1 close to t close. To earn return[t], the position
# must have been opened BEFORE t — but the signal that determines that
# position is not available until AFTER t closes.
#
# The fix is signal.shift(1) * returns, so that the position decided at
# the close of t is applied to the return earned on day t+1.
# -----------------------------------------------------------------------
strategy_returns = signal * returns

cumulative = (1 + strategy_returns).cumprod()
sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
print(f"Sharpe: {sharpe:.2f}")
print(f"Total return: {cumulative.iloc[-1] - 1:.2%}")
