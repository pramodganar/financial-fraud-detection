import os
import sys

import joblib
import streamlit as st

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "src"))

from predict import predict_transaction  # noqa: E402

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="centered",
)

st.markdown("""
<style>
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "models", "fraud_detection_model.pkl")
    return joblib.load(model_path)


model = load_model()

st.sidebar.title("Project Information")
st.sidebar.markdown("""
**Model:** XGBoost Classifier

**Techniques:**
- SMOTE (training set only)
- Balance-difference feature engineering
- Threshold tuning

**Key metrics (held-out test set):**
- PR-AUC (primary — imbalanced data)
- ROC-AUC, Precision, Recall, F1

**Supported transaction types:**
CASH_IN · CASH_OUT · DEBIT · PAYMENT · TRANSFER
""")
st.sidebar.markdown("---")
st.sidebar.info("Portfolio project — Data Scientist career transition")

st.title("Financial Fraud Detection System")
st.markdown(
    "Real-time fraud probability scoring powered by XGBoost trained "
    "on the [PaySim synthetic dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)."
)

st.header("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Transaction Step (Hour)", min_value=1, value=1)
    transaction_type = st.selectbox(
        "Transaction Type",
        ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
    )
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=1_000.0)
    oldbalanceOrg = st.number_input("Sender Old Balance ($)", min_value=0.0, value=5_000.0)

with col2:
    newbalanceOrig = st.number_input("Sender New Balance ($)", min_value=0.0, value=4_000.0)
    oldbalanceDest = st.number_input("Receiver Old Balance ($)", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("Receiver New Balance ($)", min_value=0.0, value=1_000.0)
    isFlaggedFraud = st.selectbox("System Flagged?", [0, 1])

if st.button("Predict Fraud"):
    transaction = {
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "nameOrig": "C00000",
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "nameDest": "M00000",
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFlaggedFraud": isFlaggedFraud,
    }

    result = predict_transaction(transaction, model)
    prob_pct = result["fraud_probability"] * 100

    st.markdown("---")
    st.subheader("Prediction Result")

    if result["is_fraud"]:
        st.error("Fraudulent Transaction Detected")
    else:
        st.success("Legitimate Transaction")

    st.write(f"**Fraud Probability:** {prob_pct:.2f}%")
    st.progress(min(int(prob_pct), 100))

    risk = result["risk_level"]
    if risk == "High":
        st.error(f"Risk Level: {risk}")
    elif risk == "Medium":
        st.warning(f"Risk Level: {risk}")
    else:
        st.success(f"Risk Level: {risk}")

    st.subheader("Transaction Summary")
    st.table({
        "Field": ["Type", "Amount", "Fraud Probability", "Threshold", "Risk Level"],
        "Value": [
            transaction_type,
            f"${amount:,.2f}",
            f"{prob_pct:.2f}%",
            "0.30",
            risk,
        ],
    })

    st.subheader("Fraud Pattern Flags")
    flags = []
    if amount > 200_000:
        flags.append("Large transaction amount (> $200,000)")
    if transaction_type in ("TRANSFER", "CASH_OUT"):
        flags.append("Transaction type strongly associated with fraud (TRANSFER / CASH_OUT)")
    if oldbalanceOrg > 0 and newbalanceOrig == 0:
        flags.append("Sender balance drained to zero — common in account takeover fraud")
    if isFlaggedFraud == 1:
        flags.append("Transaction pre-flagged by the payment system")

    if flags:
        for flag in flags:
            st.warning(flag)
    else:
        st.info("No specific fraud patterns detected in this transaction.")

st.markdown("---")
st.caption(
    "XGBoost · scikit-learn · imbalanced-learn · Streamlit · SHAP  |  "
    "Dataset: PaySim Synthetic Financial Transactions"
)
