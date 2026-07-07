# Baseline metrics

Train: first 4M rows. Eval: next 1M rows. Metrics on the eval split (T/CO only).
Accuracy intentionally omitted — meaningless at 0.13% positives.

Eval no-skill PR-AUC floor (prevalence) = 0.00126.

| model | PR-AUC | ROC-AUC | precision@0.5 | recall@0.5 | F1@0.5 |
|-------|--------|---------|---------------|------------|--------|
| DummyClassifier (stratified) | 0.0013 | 0.499 | 0.000 | 0.000 | 0.000 |
| LogisticRegression (balanced) | 0.491 | 0.995 | 0.035 | 0.991 | 0.067 |

Confusion (LogReg @0.5): TN=409,362 FP=14,754 FN=5 TP=532.

Read: the linear baseline already catches almost all fraud (high recall, strong
ROC/PR-AUC) but at terrible precision — ~15k false positives at the default
0.5 threshold. Threshold tuning and a stronger model (model selection and the
eval report) are about trading some of that recall for far better precision.

Caveat: 17 of the 554 eval-window frauds have a missing `type` and are dropped
by the pipeline filter, so recall here is over 537. Treated as an out-of-scope
recall ceiling; final evaluation scores the full eval set so those
count as misses.
