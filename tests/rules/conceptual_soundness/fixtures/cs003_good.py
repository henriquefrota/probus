"""Momentum factor backtest — equity universe.

Demonstrates the correct preprocessing pattern: split the data first,
then fit the scaler exclusively on the training partition. The test
partition is transformed using parameters derived only from training
observations, so no test-set statistics ever influence the model.
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
# CORRECT: Establish the temporal boundary FIRST.
#
# All preprocessing decisions (scaler parameters, feature statistics) must
# be made using only observations available before the cutoff date. This
# mirrors the information constraint faced by a live trading system.
# -----------------------------------------------------------------------
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# -----------------------------------------------------------------------
# CORRECT: Fit the scaler ONLY on training data.
#
# scaler.fit_transform(X_train) computes mean and std from training
# observations only. scaler.transform(X_test) applies those same
# parameters to the test set without re-estimating them. Test-set
# statistics are never used in the scaling computation.
# -----------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model training and evaluation ---
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
ic = np.corrcoef(y_test, y_pred)[0, 1]

print(f"MSE (OOS): {mse:.6f}")
print(f"IC  (OOS): {ic:.4f}")
