# ============================================
# IMPORT LIBRARIES
# ============================================

import os
import pandas as pd
import numpy as np
import joblib

from data_preprocessing import preprocess_data


# ============================================
# BASE DIRECTORY
# ============================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)


# ============================================
# MODEL PATHS
# ============================================

model_path = os.path.join(
    BASE_DIR,
    'models',
    'fraud_detection_model.pkl'
)

scaler_path = os.path.join(
    BASE_DIR,
    'models',
    'scaler.pkl'
)


# ============================================
# LOAD MODEL FILES
# ============================================

print("====================================")
print("LOADING MODEL FILES")
print("====================================")

model = joblib.load(
    model_path
)

scaler = joblib.load(
    scaler_path
)

print("Model loaded successfully")


# ============================================
# SAMPLE TRANSACTION DATA
# ============================================

print("\n====================================")
print("CREATING SAMPLE TRANSACTION")
print("====================================")

input_data = {
    'step': [1],
    'type': ['CASH_OUT'],
    'amount': [800000],
    'nameOrig': ['C12345'],
    'oldbalanceOrg': [900000],
    'newbalanceOrig': [0],
    'nameDest': ['M67890'],
    'oldbalanceDest': [0],
    'newbalanceDest': [800000],
    'isFlaggedFraud': [1]
}

df = pd.DataFrame(
    input_data
)

print("Sample transaction created")


# ============================================
# PREPROCESS DATA
# ============================================

print("\n====================================")
print("PREPROCESSING DATA")
print("====================================")

df = preprocess_data(
    df
)

print("Preprocessing completed")


# ============================================
# ALIGN FEATURES
# ============================================

print("\n====================================")
print("ALIGNING MODEL FEATURES")
print("====================================")

model_features = model.get_booster().feature_names

for col in model_features:

    if col not in df.columns:

        df[col] = 0

df = df[model_features]

print("Feature alignment completed")


# ============================================
# SCALE FEATURES
# ============================================

print("\n====================================")
print("SCALING FEATURES")
print("====================================")

scaled_data = scaler.transform(
    df
)

print("Feature scaling completed")


# ============================================
# FRAUD PROBABILITY
# ============================================

print("\n====================================")
print("PREDICTING FRAUD")
print("====================================")

probability = model.predict_proba(
    df
)[:,1]

fraud_probability = probability[0]

fraud_probability_percent = (
    fraud_probability * 100
)

print(
    f"Fraud Probability: {fraud_probability_percent:.2f}%"
)


# ============================================
# CUSTOM THRESHOLD
# ============================================

threshold = 0.10

prediction = int(
    fraud_probability >= threshold
)


# ============================================
# DISPLAY RESULTS
# ============================================

print("\n====================================")
print("PREDICTION RESULT")
print("====================================")

if prediction == 1:

    print(
        "⚠️ Fraudulent Transaction Detected"
    )

else:

    print(
        "✅ Legitimate Transaction"
    )


# ============================================
# RISK LEVEL
# ============================================

print("\n====================================")
print("RISK ANALYSIS")
print("====================================")

if prediction == 1:

    if fraud_probability_percent > 70:

        risk_level = "High Risk"

    else:

        risk_level = "Medium Risk"

else:

    risk_level = "Low Risk"

print(f"Risk Level: {risk_level}")


# ============================================
# FRAUD INSIGHTS
# ============================================

print("\n====================================")
print("FRAUD INSIGHTS")
print("====================================")

if input_data['amount'][0] > 200000:

    print(
        "- Large transaction amount detected"
    )

if input_data['type'][0] in ['TRANSFER', 'CASH_OUT']:

    print(
        "- Suspicious transaction type detected"
    )

if (
    input_data['oldbalanceOrg'][0] > 0
    and
    input_data['newbalanceOrig'][0] == 0
):

    print(
        "- Sender balance became zero"
    )

if input_data['isFlaggedFraud'][0] == 1:

    print(
        "- Transaction already flagged as suspicious"
    )


# ============================================
# FINAL SUMMARY
# ============================================

print("\n====================================")
print("FINAL SUMMARY")
print("====================================")

summary_df = pd.DataFrame({
    'Metric': [
        'Transaction Type',
        'Transaction Amount',
        'Fraud Probability',
        'Threshold Used',
        'Prediction',
        'Risk Level'
    ],
    'Value': [
        input_data['type'][0],
        f"${input_data['amount'][0]:,.2f}",
        f"{fraud_probability_percent:.2f}%",
        threshold,
        (
            "Fraud"
            if prediction == 1
            else
            "Legitimate"
        ),
        risk_level
    ]
})

print(summary_df)


# ============================================
# COMPLETED
# ============================================

print("\n====================================")
print("PREDICTION PIPELINE COMPLETED")
print("====================================")