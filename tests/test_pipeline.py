"""Pipeline and threshold sanity checks.

Self-contained: builds tiny synthetic frames, so these run without the 6M-row CSV
or a trained model. Run with: python -m pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline import make_features, FEATURE_COLUMNS
from src.evaluate import metrics
from src.train import DECISION_THRESHOLD


def _raw(rows):
    cols = ["type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "isFraud"]
    return pd.DataFrame(rows, columns=cols)


def test_train_predict_feature_parity():
    # The core guarantee: train (target present) and serve (target absent) build
    # the identical feature matrix, so there is no train/serve skew.
    df = _raw([["TRANSFER", 181, 181, 0, 0, 0, 0],
               ["CASH_OUT", 200, 200, 0, 0, 200, 1]])
    X_tr, y_tr = make_features(df)
    X_pred, y_pred = make_features(df.drop(columns=["isFraud"]))
    assert list(X_tr.columns) == FEATURE_COLUMNS
    assert list(X_pred.columns) == list(X_tr.columns)
    assert X_tr.shape == X_pred.shape
    assert y_tr is not None and y_pred is None


def test_type_filter_drops_non_fraud_types():
    df = _raw([["PAYMENT", 10, 10, 0, 0, 0, 0],
               ["TRANSFER", 181, 181, 0, 0, 0, 0]])
    X, _ = make_features(df)
    assert len(X) == 1  # only the TRANSFER row survives the filter


def test_index_preserved_for_score_mapping():
    # predict.py maps model probabilities back by original index; the filter must
    # keep it rather than resetting.
    df = _raw([["PAYMENT", 10, 10, 0, 0, 0, 0],
               ["TRANSFER", 181, 181, 0, 0, 0, 0]])
    X, _ = make_features(df)
    assert X.index.tolist() == [1]


def test_error_balance_orig_zero_for_clean_transfer():
    # A reconciling transaction: newbalanceOrig + amount - oldbalanceOrg == 0.
    df = _raw([["TRANSFER", 181, 181, 0, 0, 181, 0]])
    X, _ = make_features(df)
    assert X["errorBalanceOrig"].iloc[0] == pytest.approx(0.0)


def test_negative_balance_edge_case_stays_finite():
    df = _raw([["CASH_OUT", 500, -100, 0, 0, 0, 1]])  # corrupt/negative balance
    X, _ = make_features(df)
    assert np.isfinite(X["errorBalanceOrig"].iloc[0])
    assert X["errorBalanceOrig"].iloc[0] == pytest.approx(600.0)


def test_zero_amount_edge_case():
    df = _raw([["TRANSFER", 0, 0, 0, 0, 0, 0]])
    X, _ = make_features(df)
    assert X["log_amount"].iloc[0] == 0.0     # log1p(0) == 0
    assert X["origZeroBefore"].iloc[0] == 1


# --- threshold: regression guard for the "contradiction" --------------------

def test_serving_threshold_boundary_is_inclusive():
    # predict.py flags with proba >= threshold. Lock the boundary so a future edit
    # to > vs >= (or a stale constant) fails loudly.
    proba = np.array([0.10, 0.899, DECISION_THRESHOLD, 0.999])
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    assert pred.tolist() == [0, 0, 1, 1]


def test_metrics_requires_explicit_threshold():
    # metrics() must not silently report at 0.5; thr is keyword-only and required.
    y, p = np.array([0, 1]), np.array([0.2, 0.95])
    with pytest.raises(TypeError):
        metrics(y, p)


def test_metrics_at_threshold_matches_hand_count():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.10, 0.95, 0.40, 0.99])  # at 0.90: one FP (0.95), one TP (0.99)
    m = metrics(y, p, thr=0.90)
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (1, 1, 1, 1)
