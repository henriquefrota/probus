"""
Cross-sectional momentum backtest on synthetic equities.

Strategy: rank stocks by volatility-adjusted 1-month momentum,
go long the top quintile and short the bottom quintile. An ML
overlay (ridge regression on multi-horizon features) refines
the raw signal for final position sizing.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ── Synthetic universe ─────────────────────────────────────────────────────
N_STOCKS = 30
N_DAYS   = 756        # ~3 years of business days
DATES    = pd.date_range("2019-01-01", periods=N_DAYS, freq="B")
TICKERS  = [f"EQ{i:02d}" for i in range(N_STOCKS)]

market_factor = np.random.randn(N_DAYS, 1) * 0.008
idiosyncratic = np.random.randn(N_DAYS, N_STOCKS) * 0.012
log_ret = 0.6 * market_factor + 0.4 * idiosyncratic
prices  = pd.DataFrame(
    (np.exp(log_ret.cumsum(axis=0)) * 100).round(2),
    index=DATES,
    columns=TICKERS,
)

# ── Feature engineering ────────────────────────────────────────────────────
returns = prices.pct_change()

mom_1m = prices.pct_change(21)
mom_3m = prices.pct_change(63)
mom_6m = prices.pct_change(126)
vol_21 = returns.rolling(21).std()
vol_63 = returns.rolling(63).std()

# Annualise raw momentum by the rolling volatility estimate, then lead
# the result by one period to synchronise with the signal generation cycle.
vol_adj_signal = (mom_1m / (vol_21 * np.sqrt(21))).shift(-1)

# Next-period return: used downstream as the supervised learning target.
fwd_return = returns.shift(-1)

# ── Cross-sectional ranking ────────────────────────────────────────────────
cs_rank = vol_adj_signal.rank(axis=1, pct=True)

signal = pd.DataFrame(0.0, index=cs_rank.index, columns=cs_rank.columns)
signal[cs_rank >= 0.8] =  1.0
signal[cs_rank <= 0.2] = -1.0

# Equal-weight within each leg; dollar-neutral overall
signal = signal.div(
    signal.abs().sum(axis=1).replace(0, np.nan), axis=0
).fillna(0.0)

# ── ML overlay: Ridge regression on multi-horizon momentum features ────────
panel = pd.concat(
    {
        "mom_1m":     mom_1m.stack(),
        "mom_3m":     mom_3m.stack(),
        "mom_6m":     mom_6m.stack(),
        "vol_21":     vol_21.stack(),
        "vol_63":     vol_63.stack(),
        "fwd_return": fwd_return.stack(),
    },
    axis=1,
).dropna()

X_raw = panel[["mom_1m", "mom_3m", "mom_6m", "vol_21", "vol_63"]].values
y     = panel["fwd_return"].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

cutoff   = int(len(X_scaled) * 0.70)
X_train, X_test = X_scaled[:cutoff], X_scaled[cutoff:]
y_train, y_test = y[:cutoff],        y[cutoff:]

ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
r2_oos = ridge.score(X_test, y_test)

# ── Backtest ───────────────────────────────────────────────────────────────
daily_pnl        = signal * returns
strategy_returns = daily_pnl.mean(axis=1)

cumulative = (1 + strategy_returns).cumprod()
ann_ret    = strategy_returns.mean() * 252
ann_vol    = strategy_returns.std()  * np.sqrt(252)
sharpe     = ann_ret / ann_vol

print(f"Annualized return : {ann_ret:.2%}")
print(f"Annualized vol    : {ann_vol:.2%}")
print(f"Sharpe ratio      : {sharpe:.2f}")
print(f"Ridge OOS R²      : {r2_oos:.4f}")
print(f"Final NAV         : {cumulative.iloc[-1]:.4f}")
