"""Minimal Flask service around the fraud model.

POST /predict accepts one transaction record or many, and returns a fraud
probability and decision for each. Scoring goes through the same score_records /
make_features path as the batch CLI.

Run:
    python app.py
Then:
    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
         -d '{"type":"TRANSFER","amount":181.0,"oldbalanceOrg":181.0,
              "newbalanceOrig":0.0,"oldbalanceDest":0.0,"newbalanceDest":0.0}'
"""

import pandas as pd
from flask import Flask, jsonify, request

from src.predict import load_artifact, score_records

app = Flask(__name__)
ARTIFACT = load_artifact()  # loaded once at startup

# Columns make_features needs from each record.
REQUIRED = ["type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest"]


@app.get("/health")
def health():
    return jsonify(status="ok", threshold=ARTIFACT["threshold"],
                   n_features=len(ARTIFACT["features"]))


@app.post("/predict")
def predict():
    payload = request.get_json(force=True)

    # Accept a single record, a bare list, or {"records": [...]}.
    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    elif isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        return jsonify(error="send a record object or a list of records"), 400

    df = pd.DataFrame(records)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return jsonify(error=f"missing fields: {missing}"), 400

    scored = score_records(df, ARTIFACT)
    results = scored[["fraud_probability", "isFraud_pred"]].to_dict(orient="records")
    return jsonify(results=results)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
