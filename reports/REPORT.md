# Financial Fraud Detection — Report

Binary classification on ~6.36M mobile-money transactions to flag fraud
(`isFraud`). Positives are ~0.13% of rows, so the whole project is built around
precision/recall, not accuracy.

The data is split in time order, as prescribed: first 4M rows for training, the
next 1M for evaluation/threshold tuning, and the final ~1.36M held out as a
production/live set scored exactly once (see the end of this report).

## Model selected

**XGBoost** (`XGBClassifier`, `tree_method="hist"`, `n_estimators=400`,
`max_depth=6`, `learning_rate=0.1`, `subsample=0.9`, `colsample_bytree=0.9`,
`scale_pos_weight = neg/pos`).

I compared three classifiers on the eval split, all handling the imbalance with
weights rather than SMOTE (resampling 6M rows is wasteful and distorts the
probability calibration I rely on for threshold tuning):

| model | PR-AUC | ROC-AUC |
|-------|--------|---------|
| LogisticRegression (balanced) | 0.476 | 0.976 |
| HistGradientBoosting (balanced) | 0.9657 | 0.978 |
| **XGBoost** (scale_pos_weight) | **0.9659** | 0.978 |

The two boosted models tie on ranking (the PR-AUC gap is noise). I picked XGBoost
because it was much better calibrated at the default threshold (21 false positives
on eval vs 326 for HGB) and `scale_pos_weight` is a clean imbalance handle.
Missing balances are median-imputed (train-fit) in the shared pipeline before any
model sees them. HistGradientBoosting is an equal
scikit-learn-only fallback. Logistic regression ranks acceptably but its
precision is far worse; it stays only as a baseline reference.

## Features selected

All cleaning and feature engineering live in one shared module
(`src/pipeline.py`) that both training and serving import, so there is no
train/serve skew. It is a stateless function `df -> (X, y)`; the only fitted step
(median imputation) is trained on the training split and persisted with the
model.

The data only ever contains fraud in `TRANSFER` and `CASH_OUT` transactions, so
the pipeline filters to those two types; everything else is scored 0 by domain
rule without touching the model.

The 13 features:

- **Balance-reconciliation errors** — the core signal:
  - `errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg`
  - `errorBalanceDest = oldbalanceDest + amount - newbalanceDest`
  Raw balances are inconsistent in this data (they fail to reconcile 64–77% of
  the time), so the *error* is the legitimate signal, not the raw balances.
- **Zero-balance flags**: `origZeroBefore/After`, `destZeroBefore/After` — fraud
  characteristically empties the origin account.
- **Amount**: `amount` and `log_amount` (the distribution is heavily skewed).
- **`isTransfer`**: TRANSFER vs CASH_OUT.
- The four raw balance columns are kept too (available at scoring time, not
  target-derived), but they add little once the errors are present.

Dropped: `isMerchantDest` (zero variance — there are no merchant destinations
inside TRANSFER/CASH_OUT) and `isFlaggedFraud` (the business >200k rule fires
only 16 times in 6.36M rows and misses 8,197 of 8,213 frauds — a documented
curiosity, not a usable feature).

## Feature importance

Permutation importance (drop in PR-AUC when a column is shuffled), confirmed by
XGBoost's native gain:

| feature | permutation importance |
|---------|------------------------|
| errorBalanceOrig | 0.934 |
| origZeroAfter | 0.196 |
| oldbalanceOrg | 0.025 |
| amount | 0.007 |
| (rest) | < 0.005 |

`errorBalanceOrig` carries almost the entire model: fraud drains the origin
account to zero, so its reconciliation error collapses to ~0 and all but
identifies the transaction. `origZeroAfter` captures the same "account emptied"
pattern. The balance-error engineering is what makes this model work. (Note:
permutation importance understates correlated features, so `errorBalanceDest`
scores low despite separating the classes well in EDA — it is redundant with
`errorBalanceOrig`, not useless.) Plot: `reports/feature_importance.png`.

## Evaluation

On the 1M eval split (scored in full — frauds in unknown-type rows count as
misses):

- PR-AUC **0.966**, ROC-AUC **0.978**.

**Threshold = 0.90.** Recall is essentially flat across thresholds (~0.962): the
model catches every scoreable fraud but a handful, and there is a hard ceiling of
17 eval frauds with a missing `type` that cannot be scored. So the threshold is
really a false-alarm-volume decision, not a recall trade-off. F1 technically
peaks near 0.998 with zero false positives on eval, but that cutoff is overfit to
this period's calibration and won't generalize; 0.90 keeps precision at 0.980
(~11 false alarms per 1M) while staying robust. At 0.90 on eval: precision 0.980,
recall 0.962, F1 0.971. Precision-recall curve: `reports/pr_curve_eval.png`.

On why the numbers are this high and not a leak: every feature is computed from
fields available at scoring time (amounts and the pre/post balances on the
transaction itself), nothing target-derived or aggregated across other rows, and
the imputer is fit on train only. The separation is real — in this dataset fraud
empties the origin account, which collapses `errorBalanceOrig` to ~0, so one
honestly-constructed feature does most of the work. The held-out production result
below (scored once) confirms it holds out of sample rather than reflecting a leak.

A caveat worth stating: the prescribed time split is not class-stationary — fraud
density rises from 0.085% (train) to 0.314% (production). I kept the split because
it mirrors real deployment over time.

## Production / live-set result (scored once)

The held-out final ~1.36M rows, never seen during training, selection, or tuning,
scored at threshold 0.90:

| metric | value |
|--------|-------|
| rows | 1,362,620 |
| frauds | 4,278 |
| PR-AUC | 0.9634 |
| ROC-AUC | 0.9755 |
| precision | 0.998 |
| recall | 0.957 |
| F1 | 0.977 |
| confusion | TN=1,358,334  FP=8  FN=186  TP=4,092 |

This holds up against the eval numbers despite the higher fraud density. Of the
186 missed frauds, **148 are unknown-type transactions the model cannot score by
design** — so on *scoreable* traffic the effective recall is 4,092 / 4,130 =
**0.991**, at 0.998 precision (only 8 false positives across 1.36M transactions).
The missing-type rows are the real ceiling here, not the model.

## Code

- Data loader (CSV now, swappable to the `fraud_detection` table in
  `Classification.db`): `src/data.py`
- Shared cleaning + features: `src/pipeline.py`
- EDA: `notebooks/eda.ipynb` (findings in `reports/EDA_FINDINGS.md`)
- Baseline / model selection / evaluation / importance:
  `src/baseline.py`, `src/model_selection.py`, `src/eval_report.py`,
  `src/feature_importance.py`
- Train / batch scoring / API: `src/train.py`, `src/predict.py`, `app.py`
- Production scoring: `src/production_eval.py`
