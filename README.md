# 💳 Financial Fraud Detection System

An end-to-end Machine Learning project for detecting fraudulent financial transactions using XGBoost, Streamlit, and advanced fraud analytics techniques.

---

# 🚀 Project Overview

This project predicts whether a financial transaction is fraudulent using Machine Learning.

The system includes:
- Data preprocessing pipeline
- Feature engineering
- Imbalanced learning handling
- XGBoost classification
- Threshold tuning
- Real-time fraud prediction
- Streamlit deployment

---

# 📌 Problem Statement

Financial fraud causes major losses for banking and fintech companies.

The objective of this project is to build an intelligent fraud detection system capable of:
- identifying suspicious transactions
- minimizing fraud losses
- providing real-time fraud risk analysis

---

# 📂 Dataset Information

Dataset contains financial transaction records including:
- transaction type
- account balances
- transfer amounts
- fraud labels

### Features
- step
- type
- amount
- oldbalanceOrg
- newbalanceOrig
- oldbalanceDest
- newbalanceDest
- isFlaggedFraud

### Target Variable
- isFraud

---

# ⚙️ ML Pipeline

## 1. Data Preprocessing
- Missing value handling
- Infinite value handling
- One-hot encoding

## 2. Feature Engineering
Created custom fraud indicators:
- orig_balance_diff
- dest_balance_diff
- large_transaction

## 3. Imbalanced Learning
Applied:
- SMOTE
- scale_pos_weight tuning

## 4. Model Training
Models tested:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Final model selected:
- XGBoost Classifier

---

# 📊 Evaluation Metrics

| Metric | Score |
|---|---|
| Accuracy | XX |
| Precision | XX |
| Recall | XX |
| F1-Score | XX |
| ROC-AUC | XX |

> Replace XX with your actual scores.

---

# 🧠 Key ML Concepts Used

- Classification
- Imbalanced Learning
- SMOTE
- Threshold Tuning
- Feature Engineering
- Ensemble Learning
- Explainable AI
- Real-Time Prediction

---

# 🌐 Streamlit Web Application

The project includes a real-time fraud detection web application built using Streamlit.

### Features
- Real-time fraud prediction
- Fraud probability scoring
- Risk classification
- Fraud analytics insights
- Interactive UI

---

# 📸 Application Screenshots

## Home Page

Add screenshot here.

## Fraud Prediction

Add screenshot here.

## Legitimate Transaction

Add screenshot here.

---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Seaborn
- SHAP

---

# 📁 Project Structure

```text
financial-fraud-detection/
│
├── data/
├── notebooks/
├── models/
├── reports/
├── screenshots/
├── src/
├── app/
├── requirements.txt
├── README.md
└── .gitignore