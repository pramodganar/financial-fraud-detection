"""Score the held-out production/live set once.

These rows (index 5,000,000 to the end, ~1.36M) were never used for training,
model selection, or threshold tuning. This is the single honest read on how the
model behaves on unseen, later-in-time traffic.
"""

from src.data import load_data
from src.predict import load_artifact
from src.evaluate import score_full, metrics

PROD_START = 5_000_000


def main():
    df = load_data()
    prod = df.iloc[PROD_START:]
    art = load_artifact()
    thr = art["threshold"]

    y_true, proba = score_full(art["model"], prod)
    m = metrics(y_true, proba, thr=thr)

    print(f"production rows={len(y_true)} pos={int(y_true.sum())} threshold={thr}")
    print(f"PR-AUC : {m['PR_AUC']:.4f}")
    print(f"ROC-AUC: {m['ROC_AUC']:.4f}")
    print(f"precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")
    print(f"confusion: TN={m['TN']} FP={m['FP']} FN={m['FN']} TP={m['TP']}")


if __name__ == "__main__":
    main()
