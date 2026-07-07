"""Baseline model.

Logistic regression with balanced class weights, plus a Dummy classifier for a
sanity floor. Trains on the first 4M rows, evaluates on the next 1M (the
prescribed time-ordered split). Accuracy is deliberately not reported — at 0.13%
positives it would read ~99.9% for a model that finds nothing.
"""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.data import load_data
from src.pipeline import make_features, build_preprocessor

TRAIN_END = 4_000_000
EVAL_END = 5_000_000


def load_splits():
    # Split on raw row order first (the split is defined on raw rows), then
    # run feature engineering on each slice independently.
    df = load_data(nrows=EVAL_END)
    X_tr, y_tr = make_features(df.iloc[:TRAIN_END])
    X_ev, y_ev = make_features(df.iloc[TRAIN_END:EVAL_END])
    return X_tr, y_tr, X_ev, y_ev


def report(name, y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    print(f"\n--- {name} (threshold={thr}) ---")
    print(f"PR-AUC : {average_precision_score(y_true, proba):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, proba):.4f}")
    print(f"precision: {precision_score(y_true, pred, zero_division=0):.4f}")
    print(f"recall   : {recall_score(y_true, pred, zero_division=0):.4f}")
    print(f"f1       : {f1_score(y_true, pred, zero_division=0):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    print(f"confusion: TN={tn} FP={fp} FN={fn} TP={tp}")


def main():
    X_tr, y_tr, X_ev, y_ev = load_splits()
    print(f"train rows={len(X_tr)} pos={int(y_tr.sum())} | "
          f"eval rows={len(X_ev)} pos={int(y_ev.sum())}")
    print(f"eval prevalence (no-skill PR-AUC floor): {y_ev.mean():.5f}")

    # Reference floor: predicts by class frequency, learns nothing useful.
    dummy = DummyClassifier(strategy="stratified", random_state=0)
    dummy.fit(X_tr, y_tr)
    report("DummyClassifier", y_ev, dummy.predict_proba(X_ev)[:, 1])

    # Baseline: impute -> scale -> logistic regression with balanced weights.
    logit = make_pipeline(
        build_preprocessor(),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    logit.fit(X_tr, y_tr)
    report("LogisticRegression (balanced)", y_ev, logit.predict_proba(X_ev)[:, 1])


if __name__ == "__main__":
    main()
