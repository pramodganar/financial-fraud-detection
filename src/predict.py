"""Score transactions with the trained model.

Batch CLI:
    python -m src.predict --input data/raw/fraud_detection.csv --output scored.csv
    python -m src.predict --source sqlite --output scored.csv

The scoring functions (load_artifact / score_records) are also imported by the
Flask app so the API and the batch path behave identically.

Records of a type other than TRANSFER/CASH_OUT (or with a missing type) are
scored 0 by domain rule — fraud does not occur there — without going through the
model.
"""

import argparse

import joblib
import numpy as np
import pandas as pd

from src.data import load_data, repo_root
from src.pipeline import make_features

# Columns make_features needs from each record; everything after "type" must be
# numeric. Shared by the Flask API and the Streamlit app so both validate the
# same contract.
REQUIRED_COLUMNS = ["type", "amount", "oldbalanceOrg", "newbalanceOrig",
                    "oldbalanceDest", "newbalanceDest"]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]


def check_input(df: pd.DataFrame) -> str | None:
    """Validate a raw transaction frame before scoring.

    Returns an error message, or None if scoreable. On success the numeric
    columns are coerced in place (CSV uploads arrive as strings); a value that
    can't be coerced is an error rather than silently imputed.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return f"missing fields: {missing}"
    try:
        df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric)
    except (ValueError, TypeError) as e:
        return f"non-numeric value in a numeric field: {e}"
    return None


def load_artifact(path=None):
    path = path or (repo_root() / "models" / "fraud_model.joblib")
    return joblib.load(path)


def score_records(df_raw: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Add fraud_probability and isFraud_pred columns for every input row."""
    df = df_raw.reset_index(drop=True)
    X, _ = make_features(df)

    # Guard against any train/serve column drift before predicting.
    assert list(X.columns) == artifact["features"], "predict-time features != train-time"

    proba = np.zeros(len(df))
    if len(X):
        proba[X.index.to_numpy()] = artifact["model"].predict_proba(X)[:, 1]

    out = df.copy()
    out["fraud_probability"] = proba
    out["isFraud_pred"] = (proba >= artifact["threshold"]).astype(int)
    return out


def main():
    ap = argparse.ArgumentParser(description="Score transactions for fraud.")
    ap.add_argument("--input", help="input CSV (defaults to the project CSV)")
    ap.add_argument("--source", choices=["csv", "sqlite"], default="csv")
    ap.add_argument("--output", required=True, help="output CSV path")
    ap.add_argument("--nrows", type=int, default=None, help="optional row cap")
    ap.add_argument("--model", help="model artifact path")
    args = ap.parse_args()

    if args.input:
        df = pd.read_csv(args.input, nrows=args.nrows)
    else:
        df = load_data(source=args.source, nrows=args.nrows)

    artifact = load_artifact(args.model)
    scored = score_records(df, artifact)
    scored.to_csv(args.output, index=False)

    flagged = int(scored["isFraud_pred"].sum())
    print(f"scored {len(scored)} rows -> {args.output} | flagged {flagged} as fraud")


if __name__ == "__main__":
    main()
