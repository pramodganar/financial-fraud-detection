"""Streamlit UI for the fraud model.

A thin front-end over the same artifact and scoring code used by the CLI and the
Flask API (src.predict.score_records / make_features), so predictions match the
rest of the pipeline. Run:  streamlit run streamlit_app.py
"""

import pandas as pd
import streamlit as st

from src.predict import check_input, load_artifact, score_records

st.set_page_config(page_title="Fraud Detection")


@st.cache_resource
def get_artifact():
    return load_artifact()


art = get_artifact()
threshold = art["threshold"]

st.title("Transaction Fraud Detection")
st.caption(
    f"XGBoost model · decision threshold {threshold:.2f} · "
    "fraud only occurs in TRANSFER / CASH_OUT (other types are scored 0 by rule)."
)

tab_single, tab_batch = st.tabs(["Single transaction", "Batch CSV"])

with tab_single:
    with st.form("txn"):
        c1, c2 = st.columns(2)
        with c1:
            ttype = st.selectbox(
                "type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
            )
            amount = st.number_input("amount", min_value=0.0, value=181.0, step=100.0)
            oldbalanceOrg = st.number_input("oldbalanceOrg", min_value=0.0, value=181.0)
            newbalanceOrig = st.number_input("newbalanceOrig", min_value=0.0, value=0.0)
        with c2:
            oldbalanceDest = st.number_input("oldbalanceDest", min_value=0.0, value=0.0)
            newbalanceDest = st.number_input("newbalanceDest", min_value=0.0, value=0.0)
        submitted = st.form_submit_button("Score transaction")

    if submitted:
        row = pd.DataFrame([{
            "type": ttype, "amount": amount,
            "oldbalanceOrg": oldbalanceOrg, "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest, "newbalanceDest": newbalanceDest,
        }])
        scored = score_records(row, art)
        prob = float(scored["fraud_probability"].iloc[0])
        pred = int(scored["isFraud_pred"].iloc[0])

        st.metric("Fraud probability", f"{prob:.4f}")
        if pred:
            st.error(f"FLAGGED as fraud (probability >= {threshold:.2f})")
        else:
            st.success("Not flagged")
        if ttype not in ("TRANSFER", "CASH_OUT"):
            st.info("This type cannot be fraud in this dataset, so it is scored 0 "
                    "without calling the model.")

with tab_batch:
    st.write("Upload a CSV with the transaction columns; get the same scoring as "
             "the batch CLI.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        err = check_input(df)
        if err:
            st.error(err)
        else:
            scored = score_records(df, art)
            flagged = int(scored["isFraud_pred"].sum())
            st.write(f"Scored **{len(scored)}** rows · flagged **{flagged}** as fraud.")
            st.dataframe(scored.head(200))
            st.download_button(
                "Download scored CSV",
                scored.to_csv(index=False).encode(),
                file_name="scored.csv", mime="text/csv",
            )
