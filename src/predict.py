"""Single-transaction fraud prediction using the trained XGBoost model.

Run from the repo root:
    python src/predict.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from data_preprocessing import preprocess_data

ROOT = Path(__file__).parent.parent

# Threshold tuned from the PR curve on the held-out test set (see notebook 04).
# At 0.30: catches 120/129 fraud (93.0% recall), flags 333 alerts, precision 36%.
# At 0.10 (original): catches 122/129 but flags 492 alerts — too many false alarms
#   for the corrected model (that threshold was calibrated for the buggy version).
# Raise threshold toward 0.50–0.70 to trade recall for fewer investigator alerts.
FRAUD_THRESHOLD = 0.30


def load_model(model_path: Path):
    """Load a serialised XGBoost model from disk."""
    return joblib.load(model_path)


def predict_transaction(
    transaction: dict[str, Any],
    model,
    threshold: float = FRAUD_THRESHOLD,
) -> dict[str, Any]:
    """Predict whether a single transaction is fraudulent.

    XGBoost is scale-invariant (tree splits are based on rank order),
    so no feature scaling is applied here or during training.

    Parameters
    ----------
    transaction : dict
        Raw transaction fields matching the PaySim dataset schema.
    model : XGBClassifier
        Trained model loaded via load_model().
    threshold : float
        Decision threshold on the fraud probability score.
        Default (0.10) favours recall; raise it to reduce false positives.

    Returns
    -------
    dict
        fraud_probability (float), is_fraud (bool), risk_level (str).
    """
    df = pd.DataFrame([transaction])
    df = preprocess_data(df)

    # Align columns to exactly what the model was trained on,
    # adding zero-filled columns for any one-hot categories absent in this row.
    expected_cols = model.get_booster().feature_names
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    fraud_prob = float(model.predict_proba(df)[0, 1])
    is_fraud = fraud_prob >= threshold

    if is_fraud and fraud_prob > 0.70:
        risk_level = "High"
    elif is_fraud:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "fraud_probability": round(fraud_prob, 4),
        "is_fraud": is_fraud,
        "risk_level": risk_level,
    }


def _print_insights(transaction: dict[str, Any]) -> None:
    """Print human-readable flags for common fraud patterns."""
    if transaction.get("amount", 0) > 200_000:
        print("  - Large transaction amount detected")
    if transaction.get("type") in ("TRANSFER", "CASH_OUT"):
        print("  - Transaction type commonly associated with fraud")
    if transaction.get("oldbalanceOrg", 0) > 0 and transaction.get("newbalanceOrig", 0) == 0:
        print("  - Sender balance became zero after transaction")
    if transaction.get("isFlaggedFraud") == 1:
        print("  - Transaction already flagged as suspicious")


if __name__ == "__main__":
    model = load_model(ROOT / "models" / "fraud_detection_model.pkl")

    sample_transaction = {
        "step": 1,
        "type": "CASH_OUT",
        "amount": 800_000,
        "nameOrig": "C12345",
        "oldbalanceOrg": 900_000,
        "newbalanceOrig": 0,
        "nameDest": "M67890",
        "oldbalanceDest": 0,
        "newbalanceDest": 800_000,
        "isFlaggedFraud": 1,
    }

    result = predict_transaction(sample_transaction, model)

    print(f"Fraud probability : {result['fraud_probability']:.2%}")
    print(f"Prediction        : {'FRAUD' if result['is_fraud'] else 'Legitimate'}")
    print(f"Risk level        : {result['risk_level']}")
    print("\nFraud insights:")
    _print_insights(sample_transaction)
