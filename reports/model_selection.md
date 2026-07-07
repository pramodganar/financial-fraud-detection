# Model selection

Trained on the first 4M rows, evaluated on the next 1M (full eval set; unknown-
type frauds count as misses). Imbalance handled with class weights /
scale_pos_weight — no SMOTE. Selection metric: PR-AUC.

| model | PR-AUC | ROC-AUC | P@0.5 | R@0.5 | F1@0.5 | FP | FN | TP |
|-------|--------|---------|-------|-------|--------|----|----|----|
| logreg (balanced)   | 0.476  | 0.976 | 0.035 | 0.960 | 0.067 | 14,754 | 22 | 532 |
| hist_gb (balanced)  | 0.9657 | 0.978 | 0.621 | 0.966 | 0.756 | 326 | 19 | 535 |
| xgboost (scale_pos_weight) | 0.9659 | 0.978 | 0.962 | 0.964 | 0.963 | 21 | 20 | 534 |

## Decision: XGBoost

The two boosted models are tied on ranking (PR-AUC differs by 0.0002, noise). I
picked XGBoost because:
- marginally best PR-AUC, and
- it is far better calibrated at the default threshold (21 false positives vs 326
  for HGB), which makes the threshold choice in Stage 6 more forgiving, and
- scale_pos_weight is a clean, cheap imbalance handle (missing balances are
  median-imputed, train-fit, in the shared pipeline before the model).

HistGradientBoosting is an essentially equal, scikit-learn-only fallback if
avoiding the XGBoost dependency ever matters.

Logistic regression ranks acceptably (ROC-AUC 0.976) but its PR-AUC and precision
are far worse — kept only as a baseline reference.

No SMOTE: resampling 6M rows is expensive and distorts probability calibration,
which we rely on for threshold tuning. Weighting achieves the same recall without
that downside.

Config: XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
subsample=0.9, colsample_bytree=0.9, scale_pos_weight=neg/pos, tree_method="hist").
