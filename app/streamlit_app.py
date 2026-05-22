# ============================================
# IMPORT LIBRARIES
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os


# ============================================
# ADD SRC PATH
# ============================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'src')
    )
)

from data_preprocessing import preprocess_data


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="centered"
)


# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

.stNumberInput input {
    border-radius: 8px;
}

.stSelectbox div[data-baseweb="select"] {
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# ============================================
# LOAD MODEL
# ============================================

model = joblib.load(
    r"C:\Users\Dell\Desktop\classification project\financial-fraud-detection\models\fraud_detection_model.pkl"
)

scaler = joblib.load(
    r"C:\Users\Dell\Desktop\classification project\financial-fraud-detection\models\scaler.pkl"
)


# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("📊 Project Information")

st.sidebar.markdown("""
## ML Model
- XGBoost Classifier

## Techniques Used
- SMOTE
- Feature Engineering
- StandardScaler
- Ensemble Learning

## Evaluation Metrics
- ROC-AUC
- Precision
- Recall
- F1-Score

## Supported Transactions
- CASH_IN
- CASH_OUT
- DEBIT
- PAYMENT
- TRANSFER
""")

st.sidebar.markdown("---")

st.sidebar.success(
    "Built for Data Scientist Career Transition"
)


# ============================================
# MAIN TITLE
# ============================================

st.title("💳 Financial Fraud Detection System")


# ============================================
# PROJECT DESCRIPTION
# ============================================

st.markdown("""
## AI-Powered Banking Fraud Detection

This machine learning application predicts whether a financial transaction is fraudulent.

### Features
- Real-time fraud prediction
- Fraud probability scoring
- Risk classification
- XGBoost ML model
- Feature-engineered fraud detection pipeline

This project demonstrates an end-to-end Machine Learning workflow for fraud analytics.
""")


# ============================================
# USER INPUT SECTION
# ============================================

st.header("📝 Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:

    step = st.number_input(
        "Transaction Step (Hour)",
        min_value=1,
        value=1
    )

    transaction_type = st.selectbox(
        "Transaction Type",
        ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0
    )

    oldbalanceOrg = st.number_input(
        "Sender Old Balance",
        min_value=0.0,
        value=5000.0
    )

with col2:

    newbalanceOrig = st.number_input(
        "Sender New Balance",
        min_value=0.0,
        value=4000.0
    )

    oldbalanceDest = st.number_input(
        "Receiver Old Balance",
        min_value=0.0,
        value=0.0
    )

    newbalanceDest = st.number_input(
        "Receiver New Balance",
        min_value=0.0,
        value=1000.0
    )

    isFlaggedFraud = st.selectbox(
        "Flagged Fraud",
        [0, 1]
    )


# ============================================
# PREDICTION BUTTON
# ============================================

if st.button("🔍 Predict Fraud"):

    # ========================================
    # CREATE INPUT DATAFRAME
    # ========================================

    input_data = {
        'step': [step],
        'type': [transaction_type],
        'amount': [amount],
        'nameOrig': ['C12345'],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'nameDest': ['M67890'],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'isFlaggedFraud': [isFlaggedFraud]
    }

    df = pd.DataFrame(input_data)

    # ========================================
    # PREPROCESS DATA
    # ========================================

    df = preprocess_data(df)

    # ========================================
    # ALIGN FEATURES
    # ========================================

    model_features = model.get_booster().feature_names

    for col in model_features:

        if col not in df.columns:

            df[col] = 0

    df = df[model_features]

    # ========================================
    # SCALE DATA
    # ========================================

    scaled_data = scaler.transform(df)

    # ========================================
    # FRAUD PROBABILITY
    # ========================================

    probability = model.predict_proba(df)[:,1]

    fraud_probability = probability[0]

    # ========================================
    # CUSTOM THRESHOLD
    # ========================================

    threshold = 0.10

    prediction = int(
        fraud_probability >= threshold
    )

    # ========================================
    # CONVERT TO PERCENTAGE
    # ========================================

    fraud_probability_percent = (
        fraud_probability * 100
    )

    # ========================================
    # DISPLAY RESULTS
    # ========================================

    st.markdown("---")

    st.subheader("📈 Prediction Result")

    if prediction == 1:

        st.error("⚠️ Fraudulent Transaction Detected")

    else:

        st.success("✅ Legitimate Transaction")

    # ========================================
    # FRAUD PROBABILITY DISPLAY
    # ========================================

    st.write(
        f"### Fraud Probability: {fraud_probability_percent:.2f}%"
    )

    st.progress(
        int(fraud_probability_percent)
    )

    # ========================================
    # RISK LEVEL
    # ========================================

    if prediction == 1:

        if fraud_probability_percent > 70:

            risk_level = "🔴 High Risk Transaction"

            st.error(risk_level)

            risk_level_summary = "High Risk"

        else:

            risk_level = "🟠 Medium Risk Transaction"

            st.warning(risk_level)

            risk_level_summary = "Medium Risk"

    else:

        risk_level = "🟢 Low Risk Transaction"

        st.success(risk_level)

        risk_level_summary = "Low Risk"

    # ========================================
    # PREDICTION SUMMARY TABLE
    # ========================================

    st.subheader("📋 Transaction Summary")

    result_df = pd.DataFrame({
        'Metric': [
            'Transaction Type',
            'Transaction Amount',
            'Fraud Probability',
            'Threshold Used',
            'Risk Level'
        ],
        'Value': [
            transaction_type,
            f"${amount:,.2f}",
            f"{fraud_probability_percent:.2f}%",
            threshold,
            risk_level_summary
        ]
    })

    st.table(result_df)

    # ========================================
    # MODEL INSIGHT
    # ========================================

    st.subheader("🧠 Model Insight")

    if amount > 200000:

        st.warning(
            "Large transaction amount detected."
        )

    if transaction_type in ['TRANSFER', 'CASH_OUT']:

        st.warning(
            "Transaction type commonly associated with fraud."
        )

    if oldbalanceOrg > 0 and newbalanceOrig == 0:

        st.warning(
            "Sender balance became zero after transaction."
        )

    if isFlaggedFraud == 1:

        st.warning(
            "Transaction flagged by system rules."
        )


# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.markdown("""
### 🚀 Tech Stack

- Python
- XGBoost
- Scikit-learn
- Streamlit
- Pandas
- SMOTE

### 📌 Project Highlights

- End-to-End ML Pipeline
- Imbalanced Classification
- Fraud Detection System
- Real-Time Prediction
- Explainable AI Ready
- Production-Style Architecture

Built with Machine Learning for Financial Fraud Analytics.
""")