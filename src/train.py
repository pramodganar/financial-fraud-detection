# ============================================
# IMPORT LIBRARIES
# ============================================

import os
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
# BASE DIRECTORY
# ============================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)


# ============================================
# DATASET PATH
# ============================================

dataset_path = os.path.join(
    BASE_DIR,
    'data',
    'raw',
    'fraud_detection.csv'
)


# ============================================
# MODEL SAVE PATHS
# ============================================

model_save_path = os.path.join(
    BASE_DIR,
    'models',
    'fraud_detection_model.pkl'
)

scaler_save_path = os.path.join(
    BASE_DIR,
    'models',
    'scaler.pkl'
)


# ============================================
# TEST FILE SAVE PATHS
# ============================================

x_test_path = os.path.join(
    BASE_DIR,
    'data',
    'processed',
    'X_test.pkl'
)

y_test_path = os.path.join(
    BASE_DIR,
    'data',
    'processed',
    'y_test.pkl'
)


# ============================================
# CREATE REQUIRED DIRECTORIES
# ============================================

os.makedirs(
    os.path.join(BASE_DIR, 'models'),
    exist_ok=True
)

os.makedirs(
    os.path.join(BASE_DIR, 'data', 'processed'),
    exist_ok=True
)


# ============================================
# LOAD DATASET
# ============================================

print("====================================")
print("LOADING DATASET")
print("====================================")

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully")

print(f"Dataset Shape: {df.shape}")


# ============================================
# SAMPLE DATA
# ============================================

print("\n====================================")
print("SAMPLING DATA")
print("====================================")

df = df.sample(
    500000,
    random_state=42
)

print(f"Sample Dataset Shape: {df.shape}")


# ============================================
# PREPROCESS DATA
# ============================================

print("\n====================================")
print("PREPROCESSING DATA")
print("====================================")

df = preprocess_data(df)

print("Preprocessing completed")


# ============================================
# CHECK MISSING VALUES
# ============================================

print("\nChecking missing values...")

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

df = df.fillna(0)

print("Missing values handled")


# ============================================
# SPLIT FEATURES & TARGET
# ============================================

print("\n====================================")
print("FEATURES & TARGET")
print("====================================")

X = df.drop(
    'isFraud',
    axis=1
)

y = df['isFraud']

print("Features and target separated")

print(f"X Shape: {X.shape}")

print(f"y Shape: {y.shape}")


# ============================================
# TRAIN TEST SPLIT
# ============================================

print("\n====================================")
print("TRAIN TEST SPLIT")
print("====================================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train-test split completed")

print(f"X_train Shape: {X_train.shape}")

print(f"X_test Shape: {X_test.shape}")


# ============================================
# HANDLE CLASS IMBALANCE
# ============================================

print("\n====================================")
print("APPLYING SMOTE")
print("====================================")

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("SMOTE applied successfully")

print("\nBalanced Class Distribution:")

print(
    y_train_smote.value_counts()
)


# ============================================
# FEATURE SCALING
# ============================================

print("\n====================================")
print("FEATURE SCALING")
print("====================================")

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

print("\n====================================")
print("CALCULATING CLASS WEIGHT")
print("====================================")

fraud_count = y_train.value_counts()[1]

non_fraud_count = y_train.value_counts()[0]

scale_pos_weight = (
    non_fraud_count / fraud_count
)

print(
    f"Scale Pos Weight: {scale_pos_weight:.2f}"
)


# ============================================
# TRAIN XGBOOST MODEL
# ============================================

print("\n====================================")
print("TRAINING XGBOOST MODEL")
print("====================================")

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

print("Model trained successfully")


# ============================================
# MAKE PREDICTIONS
# ============================================

print("\n====================================")
print("MAKING PREDICTIONS")
print("====================================")

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:,1]

print("Predictions completed")


# ============================================
# MODEL EVALUATION
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

print("\n====================================")
print("SAVING MODEL FILES")
print("====================================")

joblib.dump(
    model,
    model_save_path
)

joblib.dump(
    scaler,
    scaler_save_path
)

print("Model files saved successfully")


# ============================================
# SAVE TEST FILES
# ============================================

joblib.dump(
    X_test,
    x_test_path
)

joblib.dump(
    y_test,
    y_test_path
)

print("Test files saved successfully")


# ============================================
# FINAL MESSAGE
# ============================================

print("\n====================================")
print("TRAINING PIPELINE COMPLETED")
print("====================================")