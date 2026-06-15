"""Equity factor model on time-ordered data — random train/test split.

The dataset has a DatetimeIndex and is constructed from rolling windows and
pct_change — clear temporal structure. train_test_split without shuffle=False
randomly mixes observations across time, leaking future data into training.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

np.random.seed(42)

n = 1000
dates = pd.date_range("2018-01-01", periods=n, freq="B")
prices = pd.Series(np.random.randn(n).cumsum() + 100, index=dates)
returns = prices.pct_change().dropna()

features = pd.DataFrame(
    {
        "mom_20": prices.pct_change(20).reindex(returns.index),
        "vol_20": returns.rolling(20).std(),
        "rsi": returns.rolling(14).mean() / (returns.rolling(14).std() + 1e-9),
    },
    index=returns.index,
).dropna()

y = returns.shift(-1).reindex(features.index).dropna()
X = features.reindex(y.index).values

X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.3)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

oos_r2 = model.score(X_test, y_test)
ic = np.corrcoef(model.predict(X_test), y_test)[0, 1]

print(f"OOS R²: {oos_r2:.4f}")
print(f"OOS IC: {ic:.4f}")
