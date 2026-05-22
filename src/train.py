# ============================================
# IMPORT LIBRARIES
# ============================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

from data_preprocessing import preprocess_data


# ============================================
# LOAD DATASET
# ============================================

print("Loading dataset...")

df = pd.read_csv(
    r"C:\Users\Dell\Desktop\classification project\financial-fraud-detection\data\raw\fraud_detection.csv"
)

print("Dataset loaded successfully")


# ============================================
# SAMPLE DATA
# ============================================

print("Creating sample dataset...")

df = df.sample(
    500000,
    random_state=42
)

print(f"Sample dataset shape: {df.shape}")


# ============================================
# PREPROCESS DATA
# ============================================

print("Preprocessing dataset...")

df = preprocess_data(df)

print("Preprocessing completed")


# ============================================
# SPLIT FEATURES & TARGET
# ============================================

X = df.drop('isFraud', axis=1)

y = df['isFraud']

print("Features and target separated")


# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train-test split completed")

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# ============================================
# HANDLE CLASS IMBALANCE
# ============================================

print("Applying SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("SMOTE completed")

print("\nBalanced class distribution:")

print(y_train_smote.value_counts())


# ============================================
# FEATURE SCALING
# ============================================

print("\nApplying StandardScaler...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_smote
)

X_test_scaled = scaler.transform(
    X_test
)

print("Feature scaling completed")


# ============================================
# CALCULATE CLASS WEIGHT
# ============================================

fraud_count = y_train.value_counts()[1]

non_fraud_count = y_train.value_counts()[0]

scale_pos_weight = (
    non_fraud_count / fraud_count
)

print(f"\nScale Pos Weight: {scale_pos_weight:.2f}")


# ============================================
# TRAIN IMPROVED XGBOOST MODEL
# ============================================

print("\nTraining Improved XGBoost Model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

model.fit(
    X_train_smote,
    y_train_smote
)

print("XGBoost model trained successfully")


# ============================================
# MAKE PREDICTIONS
# ============================================

print("\nMaking predictions...")

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:,1]

print("Predictions completed")


# ============================================
# EVALUATE MODEL
# ============================================

print("\n====================================")
print("MODEL EVALUATION")
print("====================================")

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================
# CLASSIFICATION REPORT
# ============================================

print("\n====================================")
print("CLASSIFICATION REPORT")
print("====================================")

print(
    classification_report(
        y_test,
        predictions
    )
)


# ============================================
# SAVE MODEL
# ============================================

print("\nSaving model and scaler...")

joblib.dump(
    model,
    r"C:\Users\Dell\Desktop\classification project\financial-fraud-detection\models\fraud_detection_model.pkl"
)

joblib.dump(
    scaler,
    r"C:\Users\Dell\Desktop\classification project\financial-fraud-detection\models\scaler.pkl"
)

print("Model saved successfully")


# ============================================
# SAVE TEST FILES
# ============================================

joblib.dump(
    X_test,
    r"C:\Users\Dell\Desktop\classification project\financial-fraud-detection\data\processed\X_test.pkl"
)

joblib.dump(
    y_test,
    r"C:\Users\Dell\Desktop\classification project\financial-fraud-detection\data\processed\y_test.pkl"
)

print("Test files saved successfully")


# ============================================
# FINAL MESSAGE
# ============================================

print("\n====================================")
print("TRAINING PIPELINE COMPLETED")
print("====================================")