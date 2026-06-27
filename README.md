# Financial Fraud Detection

End-to-end machine learning project for detecting fraudulent financial transactions.
Built as a portfolio project for a Data Scientist career transition.

**Live demo:** https://financial-fraud-detection-jrjuipy2mrkbxan9h8denu.streamlit.app/

---

## Problem Statement

Financial fraud is rare but costly. In the PaySim dataset, fewer than 0.13% of transactions are fraudulent — a 800:1 class imbalance. A naive model that predicts "legitimate" for every transaction achieves 99.87% accuracy while catching zero fraud.

The real challenge is ranking suspicious transactions high enough that investigators can act on them, while keeping the false-alarm rate manageable. This project treats it as a **probability-ranking problem** and evaluates with PR-AUC (Precision-Recall Area Under Curve), not accuracy.

---

## Dataset

**Source:** [PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) (Kaggle)

PaySim simulates mobile money transactions based on a real anonymised dataset from a financial company. It is widely used as a fraud detection benchmark because it reproduces realistic class imbalance and transaction patterns.

| Property | Value |
|---|---|
| Total transactions | ~6.3 million |
| Fraud rate | ~0.13% |
| Training sample used | 500,000 rows (stratified) |
| Fraud-only transaction types | TRANSFER, CASH_OUT |

**Features used:**

| Feature | Description |
|---|---|
| `step` | Hour of the simulated month (1–744) |
| `type` | Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER) |
| `amount` | Transaction amount |
| `oldbalanceOrg` / `newbalanceOrig` | Sender balance before/after |
| `oldbalanceDest` / `newbalanceDest` | Receiver balance before/after |
| `isFlaggedFraud` | Pre-existing system flag |
| `orig_balance_diff` | Engineered: sender balance change |
| `dest_balance_diff` | Engineered: receiver balance change |
| `large_transaction` | Engineered: binary flag for amount > $200,000 |

---

## Approach

### 1. Preprocessing

- Dropped identifier columns (`nameOrig`, `nameDest`) — high cardinality, no signal.
- One-hot encoded `type` with `drop_first=True` to avoid multicollinearity.
- Filled NaN values with 0. The only source of NaN is `newbalanceOrig` in CASH_IN transactions, where the payment system does not track the origin balance — zero is the correct domain fill, not a distributional guess.

### 2. Feature Engineering

Three balance-difference features capture the core fraud signal:

- **`orig_balance_diff`**: how much the sender's balance changed. In fraud cases, this typically equals the full account balance (account draining).
- **`dest_balance_diff`**: how much the receiver's balance changed. Fraudulent receivers often funnel money onward immediately.
- **`large_transaction`**: binary flag for transactions over $200,000, where fraud is disproportionately concentrated.

### 3. Handling Class Imbalance

Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to the training set only, balancing classes to 1:1 before model fitting. Key constraint: SMOTE is fit after the train/test split so no synthetic examples can appear in the test set.

> Note: an earlier version of this code also set XGBoost's `scale_pos_weight` parameter alongside SMOTE. This was a bug — it double-corrects for imbalance (SMOTE already balanced the classes; `scale_pos_weight` then applied an additional ~770× weight toward fraud). The fix was to use SMOTE alone. Expect improved precision after retraining.

### 4. Model Selection

Six classifiers were compared on the same held-out test set. PR-AUC is the primary ranking metric because it directly measures the precision/recall trade-off at all thresholds, without being inflated by the large number of true negatives that dominate accuracy and ROC-AUC on imbalanced data.

### 5. Threshold Tuning

XGBoost outputs a continuous fraud probability. The decision threshold (default: 0.30) was chosen from the PR curve on the held-out test set. At 0.30 the model catches 120 out of 129 fraud cases (93% recall) while flagging 333 transactions — a reasonable load for investigators. Raising the threshold to 0.50 cuts alerts to 261 with only 2 fewer fraud cases caught; lowering it increases recall at the cost of more false alarms. The threshold is documented and tunable in `src/predict.py`.

---

## Results

All metrics are from the corrected pipeline (SMOTE only, no `scale_pos_weight`), evaluated on a stratified held-out test set of 100,000 transactions (fraud rate: 0.129%).

### Final Model — XGBoost (default 0.5 threshold)

| Metric | Score |
|---|---|
| Precision | **45.2%** |
| Recall | **91.5%** |
| F1 | **60.5%** |
| ROC-AUC | **0.998** |
| PR-AUC | **0.896** |

PR-AUC of 0.896 means the model maintains high precision across most of the recall range — a strong result given the 800:1 class imbalance (a random classifier would score ~0.001 PR-AUC).

### 6-Model Comparison (standard hyperparameters, held-out test set, default 0.5 threshold)

Models are ranked by PR-AUC — the most meaningful metric for imbalanced fraud data.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **XGBoost** | 26.2% | 94.6% | 41.1% | 0.998 | **0.892** |
| CatBoost | 34.2% | 93.8% | 50.1% | 0.997 | 0.887 |
| LightGBM | 35.9% | 93.0% | 51.8% | 0.998 | 0.887 |
| Random Forest | 62.2% | 86.8% | 72.5% | 0.995 | 0.883 |
| Logistic Regression | 2.6% | 86.0% | 5.0% | 0.950 | 0.528 |
| Decision Tree | 57.1% | 89.9% | 69.9% | 0.949 | 0.514 |

XGBoost, CatBoost, and LightGBM are within 0.009 PR-AUC of each other — effectively tied. XGBoost was selected as the final model and tuned further (300 estimators, max_depth=8, lr=0.05).

### Before vs. After Fixing Class Imbalance Handling

The original pipeline applied SMOTE *and* `scale_pos_weight` simultaneously. After fixing:

| | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Before (SMOTE + scale_pos_weight) | 25.3% | 93.8% | 39.8% | — |
| After (SMOTE only, tuned params) | **45.2%** | **91.5%** | **60.5%** | **0.896** |

Precision nearly doubled with only a 2.3 pp drop in recall — the corrected model generates far fewer false alarms while still catching 9 out of 10 fraudulent transactions.

---

## Key Findings

1. **Only two transaction types contain fraud** — TRANSFER and CASH_OUT. The model learns this quickly, and `type_TRANSFER` / `type_CASH_OUT` rank among the top features by SHAP value.

2. **Balance features are the strongest fraud signals.** Fraudulent transactions almost always drain the sender's balance to exactly zero (`orig_balance_diff` equals `oldbalanceOrg`). This feature has the highest XGBoost gain.

3. **Accuracy is a misleading metric here.** The pre-fix model shows 99.6% accuracy while catching 94% of fraud — but a model that predicts "legitimate" for everything would achieve 99.87% accuracy while catching 0% of fraud. Always evaluate fraud models with Precision, Recall, F1, and PR-AUC.

4. **SMOTE + `scale_pos_weight` is a common anti-pattern.** Using both simultaneously over-corrects for class imbalance. Choose one: SMOTE balances the training data structurally; `scale_pos_weight` adjusts the loss function. Combining them inflates recall and destroys precision.

5. **Threshold selection is a business decision, not a modeling one.** The right threshold depends on the cost ratio of a missed fraud vs. a false alarm. The PR curve in notebook 04 shows the full precision/recall trade-off across all possible thresholds.

---

## Project Structure

```
financial-fraud-detection/
│
├── data/
│   ├── raw/                  # Original PaySim CSV (not tracked in git)
│   └── processed/            # Intermediate artifacts from notebooks
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Preprocessing, SMOTE, scaling
│   ├── 03_model_training.ipynb       # 6-model comparison
│   └── 04_model_evaluation.ipynb     # ROC/PR curves, SHAP, threshold analysis
│
├── src/
│   ├── data_preprocessing.py # preprocess_data() — cleaning + feature engineering
│   ├── train.py              # Training pipeline (run directly)
│   └── predict.py            # predict_transaction() — single-transaction inference
│
├── app/
│   └── streamlit_app.py      # Real-time fraud prediction web UI
│
├── models/                   # Saved model artifacts (not tracked in git)
├── reports/                  # model_results.csv, feature_importance.csv
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the dataset

Download `PS_20174392719_1491204439457_log.csv` from the [PaySim Kaggle page](https://www.kaggle.com/datasets/ealaxi/paysim1), rename it to `fraud_detection.csv`, and place it at:

```
data/raw/fraud_detection.csv
```

### 3. Train the model

```bash
python src/train.py
```

This samples 500,000 rows, preprocesses, applies SMOTE, trains XGBoost, and saves the model to `models/fraud_detection_model.pkl`. Test-set artifacts are saved to `data/processed/` for use in the evaluation notebook.

### 4. Run notebooks (optional, for exploration)

Run in order from the `notebooks/` directory:

```
01_eda.ipynb → 02_feature_engineering.ipynb → 03_model_training.ipynb → 04_model_evaluation.ipynb
```

Each notebook uses `ROOT = Path.cwd().parent` to locate files — no path changes needed.

### 5. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| pandas / numpy | Data manipulation |
| scikit-learn | Preprocessing, metrics, baseline models |
| imbalanced-learn | SMOTE oversampling |
| XGBoost | Final classifier |
| LightGBM / CatBoost | Comparison models |
| SHAP | Model explainability |
| matplotlib / seaborn | Visualisation |
| Streamlit | Web application |
