"""Compare a few classifiers suited to heavy imbalance.

Three candidates, all handling imbalance via weights (no SMOTE — resampling 6M
rows is wasteful and distorts the calibration we need for threshold tuning):
  - LogisticRegression, class_weight balanced  (linear reference)
  - HistGradientBoostingClassifier, class_weight balanced
  - XGBoost, scale_pos_weight = neg/pos

Selection metric is PR-AUC on the eval split, scored over the full 1M rows
(unknown-type frauds count as misses). Threshold-dependent numbers are at 0.5
just for context; the real threshold choice happens in Stage 6.
"""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import load_data
from src.pipeline import make_features, build_preprocessor
from src.evaluate import score_full, metrics

TRAIN_END = 4_000_000
EVAL_END = 5_000_000


def build_models(y_tr):
    pos = int(y_tr.sum())
    neg = len(y_tr) - pos
    spw = neg / pos  # xgboost imbalance handle

    return {
        "logreg": make_pipeline(
            build_preprocessor(),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        ),
        "hist_gb": make_pipeline(
            build_preprocessor(),
            HistGradientBoostingClassifier(
                class_weight="balanced", learning_rate=0.1,
                max_iter=300, random_state=0,
            ),
        ),
        "xgboost": make_pipeline(
            build_preprocessor(),
            XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9,
                scale_pos_weight=spw, eval_metric="aucpr",
                tree_method="hist", n_jobs=-1, random_state=0,
            ),
        ),
    }


def main():
    df = load_data(nrows=EVAL_END)
    train_raw = df.iloc[:TRAIN_END]
    eval_raw = df.iloc[TRAIN_END:EVAL_END]
    X_tr, y_tr = make_features(train_raw)
    print(f"train T/CO rows={len(X_tr)} pos={int(y_tr.sum())}")
    print(f"eval rows (full)={len(eval_raw)} pos={int(eval_raw.isFraud.sum())}\n")

    rows = []
    for name, model in build_models(y_tr).items():
        model.fit(X_tr, y_tr)
        y_true, proba = score_full(model, eval_raw)
        m = metrics(y_true, proba, thr=0.5)  # context only; real point is Stage 6
        rows.append((name, m))
        print(f"{name:8s} PR-AUC={m['PR_AUC']:.4f} ROC-AUC={m['ROC_AUC']:.4f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"(FP={m['FP']} FN={m['FN']} TP={m['TP']})")

    best = max(rows, key=lambda r: r[1]["PR_AUC"])
    print(f"\nbest by PR-AUC: {best[0]}")


if __name__ == "__main__":
    main()
