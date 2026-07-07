"""Evaluation helpers shared by model selection and final evaluation.

score_full is the honest scorer: it scores *every* raw row. Transactions outside
TRANSFER/CASH_OUT (including rows with a missing type) are assigned probability 0
by domain rule, so any fraud hiding in those rows correctly counts as a miss
rather than being silently dropped from the denominator.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.pipeline import make_features


def score_full(model, df_raw):
    """Return (y_true, proba) aligned to every row of df_raw."""
    df_raw = df_raw.reset_index(drop=True)
    X, _ = make_features(df_raw)
    proba = np.zeros(len(df_raw))
    proba[X.index.to_numpy()] = model.predict_proba(X)[:, 1]
    y_true = df_raw["isFraud"].to_numpy()
    return y_true, proba


def metrics(y_true, proba, *, thr):
    # thr is required and keyword-only: reporting metrics at a threshold you did
    # not choose is how the 0.90 decision point silently drifts. Callers must say
    # which operating point they mean.
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "PR_AUC": average_precision_score(y_true, proba),
        "ROC_AUC": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }
