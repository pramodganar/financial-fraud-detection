"""XGBoost fraud detection training pipeline.

Run from the repo root:
    python src/train.py
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_preprocessing import preprocess_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RANDOM_STATE = 42
SAMPLE_SIZE = 500_000
TEST_SIZE = 0.2


def load_data(path: Path) -> pd.DataFrame:
    """Sample 500k rows from the full dataset for tractable training."""
    log.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    log.info("Full dataset shape: %s", df.shape)
    sampled = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
    log.info("Sample shape: %s", sampled.shape)
    return sampled


def build_model() -> XGBClassifier:
    """Return the XGBoost classifier with tuned hyperparameters.

    Note: scale_pos_weight is intentionally omitted because SMOTE is applied
    to the training set beforehand, already balancing the classes 1:1.
    Using both SMOTE and scale_pos_weight double-corrects for imbalance,
    over-aggressively predicting fraud and destroying precision.
    """
    return XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )


def evaluate(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute classification metrics suitable for imbalanced fraud data.

    PR-AUC (average precision) is the primary metric here because:
    - Fraud is rare (~0.13% of transactions).
    - ROC-AUC can be misleadingly optimistic on skewed classes.
    - PR-AUC penalises false positives more honestly.
    """
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def main() -> None:
    data_path = ROOT / "data" / "raw" / "fraud_detection.csv"
    model_path = ROOT / "models" / "fraud_detection_model.pkl"
    x_test_path = ROOT / "data" / "processed" / "X_test.pkl"
    y_test_path = ROOT / "data" / "processed" / "y_test.pkl"

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)

    log.info("Preprocessing data...")
    df = preprocess_data(df)

    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log.info("Train shape: %s  |  Test shape: %s", X_train.shape, X_test.shape)
    log.info("Fraud rate in test set: %.4f%%", y_test.mean() * 100)

    log.info("Applying SMOTE to training set only...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    log.info("Balanced class distribution:\n%s", y_train_bal.value_counts().to_string())

    log.info("Training XGBoost model...")
    model = build_model()
    model.fit(X_train_bal, y_train_bal)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_pred, y_prob)
    log.info(
        "\nTest-set metrics (default 0.5 threshold):\n"
        "  Precision : %.4f\n"
        "  Recall    : %.4f\n"
        "  F1        : %.4f\n"
        "  ROC-AUC   : %.4f\n"
        "  PR-AUC    : %.4f  <-- primary metric for imbalanced fraud data",
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
        metrics["pr_auc"],
    )
    log.info("\nClassification report:\n%s", classification_report(y_test, y_pred))

    joblib.dump(model, model_path)
    log.info("Model saved to %s", model_path)

    # Persist test set so evaluation notebook can load the same held-out split.
    joblib.dump(X_test, x_test_path)
    joblib.dump(y_test, y_test_path)
    log.info("Test artifacts saved to data/processed/")


if __name__ == "__main__":
    main()
