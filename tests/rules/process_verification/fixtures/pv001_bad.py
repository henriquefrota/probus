"""Cross-sectional momentum factor — missing random seed.

The simulation uses numpy's random number generator throughout but never
calls np.random.seed(). Results will differ on every run, making the
reported information coefficient non-reproducible.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

n = 1000
dates = pd.date_range("2018-01-01", periods=n, freq="B")
prices = pd.Series(
    np.random.randn(n).cumsum() * 0.5 + 100,
    index=dates,
    name="close",
)
returns = prices.pct_change().dropna()

features = pd.DataFrame(
    {
        "mom_5d": prices.pct_change(5).reindex(returns.index),
        "mom_20d": prices.pct_change(20).reindex(returns.index),
        "vol_10d": returns.rolling(10).std(),
        "noise": np.random.randn(len(returns)),
    },
    index=returns.index,
).dropna()

y = returns.shift(-1).reindex(features.index).dropna()
features = features.reindex(y.index)

X_train = features.iloc[:700].values
y_train = y.iloc[:700].values
X_test = features.iloc[700:].values
y_test = y.iloc[700:].values

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

preds = model.predict(X_test)
ic = np.corrcoef(preds, y_test)[0, 1]

print(f"IC: {ic:.4f}")
