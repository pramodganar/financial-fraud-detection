"""Shared cleaning + feature engineering. Train and predict both import this so
they build the same matrix.

make_features is stateless. build_preprocessor() is the only fitted step (median
impute) — train fits and persists it, predict reloads it.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

TARGET = "isFraud"

# Fraud only ever occurs in these two types (confirmed in EDA); everything else
# is dropped because it carries no fraud signal.
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]

# Fixed column order. make_features always returns exactly these, in this order,
# so the train and predict matrices line up.
FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "isTransfer",
    "origZeroBefore",
    "origZeroAfter",
    "destZeroBefore",
    "destZeroAfter",
]


def make_features(df: pd.DataFrame):
    """Raw transactions -> (X, y). y is None when the target is absent (predict).

    Filters to TRANSFER/CASH_OUT; other (or missing) types are dropped and scored
    0 by rule at serving time. X keeps the original index so callers can map
    predictions back. Missing balances stay NaN for the imputer; the zero flags
    treat a missing balance as not-zero.
    """
    d = df[df["type"].isin(FRAUD_TYPES)].copy()

    d["log_amount"] = np.log1p(d["amount"])
    # Balance reconciliation errors. For a clean transaction these are ~0; fraud
    # breaks the reconciliation in a characteristic way (see EDA).
    d["errorBalanceOrig"] = d["newbalanceOrig"] + d["amount"] - d["oldbalanceOrg"]
    d["errorBalanceDest"] = d["oldbalanceDest"] + d["amount"] - d["newbalanceDest"]

    d["isTransfer"] = (d["type"] == "TRANSFER").astype("int8")
    d["origZeroBefore"] = (d["oldbalanceOrg"] == 0).astype("int8")
    d["origZeroAfter"] = (d["newbalanceOrig"] == 0).astype("int8")
    d["destZeroBefore"] = (d["oldbalanceDest"] == 0).astype("int8")
    d["destZeroAfter"] = (d["newbalanceDest"] == 0).astype("int8")

    X = d[FEATURE_COLUMNS]
    y = d[TARGET] if TARGET in d.columns else None
    return X, y


def build_preprocessor() -> SimpleImputer:
    """Unfitted median imputer. Output stays a DataFrame so feature names survive
    into the importance plots."""
    imp = SimpleImputer(strategy="median")
    imp.set_output(transform="pandas")
    return imp


if __name__ == "__main__":
    from src.data import load_data

    X, y = make_features(load_data(nrows=200_000))
    print("X shape:", X.shape)
    print("columns:", list(X.columns))
    print("positives:", int(y.sum()))
