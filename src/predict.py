import pandas as pd
import joblib

from data_preprocessing import preprocess_data


# ============================================
# LOAD MODEL
# ============================================

model = joblib.load(
    r"C:\Users\Dell\Desktop\classification project\financial-fraud-detection\models\fraud_detection_model.pkl"
)

scaler = joblib.load(
    r"C:\Users\Dell\Desktop\classification project\financial-fraud-detection\models\scaler.pkl"
)

print("Model loaded successfully")


# ============================================
# CREATE SAMPLE INPUT
# ============================================

sample_data = {
    'step': [1],
    'type': ['TRANSFER'],
    'amount': [500000],
    'nameOrig': ['C12345'],
    'oldbalanceOrg': [600000],
    'newbalanceOrig': [100000],
    'nameDest': ['M67890'],
    'oldbalanceDest': [0],
    'newbalanceDest': [500000],
    'isFlaggedFraud': [0]
}

df = pd.DataFrame(sample_data)


# ============================================
# PREPROCESS INPUT
# ============================================

df = preprocess_data(df)


# ============================================
# ALIGN COLUMNS
# ============================================

model_features = model.get_booster().feature_names

for col in model_features:
    if col not in df.columns:
        df[col] = 0

df = df[model_features]


# ============================================
# SCALE INPUT
# ============================================

scaled_data = scaler.transform(df)


# ============================================
# PREDICT
# ============================================

prediction = model.predict(df)

probability = model.predict_proba(df)[:,1]

print("\nPrediction Result")

if prediction[0] == 1:
    print("Fraud Transaction Detected")
else:
    print("Legitimate Transaction")

print(f"Fraud Probability: {probability[0]:.4f}")