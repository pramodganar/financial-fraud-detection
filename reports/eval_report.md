# Evaluation (XGBoost, eval split)

Trained on the first 4M rows, evaluated on the next 1M (full set; unknown-type
frauds counted as misses).

- PR-AUC : 0.9659
- ROC-AUC: 0.9781
- eval positives: 554 (of which 17 have a missing `type` and cannot be scored,
  so the recall ceiling is 537/554 = 0.969)

## Threshold choice

Recall is essentially flat across thresholds (~0.962–0.966): the model already
catches every scoreable fraud but ~4. Dropping the threshold from 0.90 to 0.10
recovers only 2 more frauds (TP 533 -> 535) while multiplying false positives
(11 -> 84), so it isn't worth it. The real lever is precision / false-alarm
volume.

| threshold | precision | recall | F1 | FP | FN | TP |
|-----------|-----------|--------|----|----|----|----|
| 0.10 | 0.864 | 0.966 | 0.912 | 84 | 19 | 535 |
| 0.30 | 0.940 | 0.964 | 0.952 | 34 | 20 | 534 |
| 0.50 | 0.962 | 0.964 | 0.963 | 21 | 20 | 534 |
| 0.70 | 0.973 | 0.964 | 0.968 | 15 | 20 | 534 |
| **0.90** | **0.980** | **0.962** | **0.971** | **11** | **21** | **533** |
| 0.9984 (F1-max) | 1.000 | 0.962 | 0.981 | 0 | 21 | 533 |

**Chosen threshold: 0.90.** F1 technically peaks near 0.998 (zero FP on eval),
but that cutoff sits at the extreme edge of the score distribution and the zero-FP
result is overfit to this period's calibration — it will not hold on the
production set. 0.90 keeps precision very high (0.980, ~11 false alarms per 1M
transactions) while staying off the razor's edge, so it should generalize. Recall
is unchanged at 0.962, right against the unknown-type ceiling.

Confusion matrix at 0.90: TN=999,435  FP=11  FN=21  TP=533.

Plot: `reports/pr_curve_eval.png`.
