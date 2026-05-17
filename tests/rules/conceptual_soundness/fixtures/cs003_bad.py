"""Momentum factor backtest — equity universe.

Demonstrates a common methodological error: fitting a feature scaler on the
full dataset before performing the train/test split. The scaler absorbs
mean and variance from the held-out test period, leaking future statistics
into the in-sample training process and inflating OOS performance estimates.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- Simulated price data ---
np.random.seed(42)
n = 2000
dates = pd.date_range("2015-01-01", periods=n, freq="B")

returns = pd.DataFrame(
    np.random.randn(n, 50) * 0.01,
    index=dates,
    columns=[f"asset_{i}" for i in range(50)],
)

# --- Factor construction ---
mom_20 = returns.rolling(20).mean()
mom_60 = returns.rolling(60).mean()
vol_20 = returns.rolling(20).std()
rsi_proxy = returns.rolling(14).mean() / (returns.rolling(14).std() + 1e-9)

features = pd.concat(
    [
        mom_20.mean(axis=1),
        mom_60.mean(axis=1),
        vol_20.mean(axis=1),
        rsi_proxy.mean(axis=1),
    ],
    axis=1,
).dropna()
features.columns = ["mom_20", "mom_60", "vol_20", "rsi"]

fwd_return = returns.mean(axis=1).shift(-1).reindex(features.index).dropna()
features = features.reindex(fwd_return.index)

X = features.values
y = fwd_return.values

# -----------------------------------------------------------------------
# BUG: The scaler is fitted on the entire dataset BEFORE the split.
#
# StandardScaler computes mean and std over all N observations, including
# the observations that will later become the test set. When those same
# observations are evaluated out-of-sample, their true distributional
# properties (mean, std) are already embedded in the scaler's parameters.
# This is data leakage: the model has implicitly "seen" the test set during
# preprocessing, which biases OOS metrics upward.
# -----------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # leaks test-set statistics into training

# Split *after* fitting — the wrong order.
split_idx = int(len(X_scaled) * 0.7)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Model training and evaluation ---
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
ic = np.corrcoef(y_test, y_pred)[0, 1]

print(f"MSE (OOS): {mse:.6f}")
print(f"IC  (OOS): {ic:.4f}")
