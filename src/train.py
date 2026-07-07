"""Fit XGBoost on the first 4M rows and persist it.

Saves the fitted pipeline (imputer + XGBoost), the decision threshold, and the
feature list to one joblib artifact for predict.py / app.py to load.
"""

import time

import joblib

from src.data import load_data, repo_root
from src.pipeline import make_features, FEATURE_COLUMNS
from src.model_selection import build_models

TRAIN_END = 4_000_000
DECISION_THRESHOLD = 0.90  # chosen in Stage 6 (eval): precision ~0.98, recall ~0.962


def model_path():
    return repo_root() / "models" / "fraud_model.joblib"


def main():
    df = load_data(nrows=TRAIN_END)
    X, y = make_features(df)
    assert list(X.columns) == FEATURE_COLUMNS, "feature columns drifted from spec"
    print(f"training on {len(X)} TRANSFER/CASH_OUT rows, {int(y.sum())} fraud")

    model = build_models(y)["xgboost"]
    t0 = time.time()
    model.fit(X, y)
    print(f"fit done in {time.time() - t0:.1f}s")

    artifact = {
        "model": model,
        "threshold": DECISION_THRESHOLD,
        "features": FEATURE_COLUMNS,
        "n_train_rows": len(X),
        "n_train_pos": int(y.sum()),
    }
    out = model_path()
    joblib.dump(artifact, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
