"""Evaluate the selected model (XGBoost) on the eval split and report the threshold.

Trains on the first 4M rows, scores the full next 1M, computes PR-AUC / ROC-AUC,
and sweeps the decision threshold. The operating point we ship is
DECISION_THRESHOLD (0.90) from train.py — the F1-max point is shown alongside as
context but not used: it sits at zero FP on this period, which is overfit to the
eval calibration and won't hold on production. Saves the PR curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from src.data import load_data, repo_root
from src.model_selection import build_models, TRAIN_END, EVAL_END
from src.pipeline import make_features
from src.evaluate import score_full, metrics
from src.train import DECISION_THRESHOLD


def main():
    df = load_data(nrows=EVAL_END)
    X_tr, y_tr = make_features(df.iloc[:TRAIN_END])
    eval_raw = df.iloc[TRAIN_END:EVAL_END]

    model = build_models(y_tr)["xgboost"]
    model.fit(X_tr, y_tr)

    y_true, proba = score_full(model, eval_raw)
    n_pos = int(y_true.sum())
    print(f"eval rows={len(y_true)} pos={n_pos}")

    prec, rec, thr = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns one fewer threshold than points; align.
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    f1_max_thr = float(thr[int(np.argmax(f1))])
    print(f"\nF1-max threshold = {f1_max_thr:.4f} (context only; we ship "
          f"{DECISION_THRESHOLD} — see report)")

    print("\nthreshold sweep:")
    print(f"{'thr':>8} {'prec':>7} {'recall':>7} {'f1':>7} {'FP':>6} {'FN':>5} {'TP':>5}")
    for t in [0.10, 0.30, 0.50, 0.70, DECISION_THRESHOLD, f1_max_thr]:
        m = metrics(y_true, proba, thr=t)
        tag = "  <- chosen" if abs(t - DECISION_THRESHOLD) < 1e-9 else ""
        print(f"{t:8.4f} {m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} "
              f"{m['FP']:6d} {m['FN']:5d} {m['TP']:5d}{tag}")

    chosen = metrics(y_true, proba, thr=DECISION_THRESHOLD)
    print("\n=== chosen operating point (thr=%.2f) ===" % DECISION_THRESHOLD)
    print(f"PR-AUC : {chosen['PR_AUC']:.4f}")
    print(f"ROC-AUC: {chosen['ROC_AUC']:.4f}")
    print(f"precision={chosen['precision']:.3f} recall={chosen['recall']:.3f} "
          f"f1={chosen['f1']:.3f}")
    print(f"confusion: TN={chosen['TN']} FP={chosen['FP']} "
          f"FN={chosen['FN']} TP={chosen['TP']}")
    print(f"recall ceiling (unknown-type frauds unscoreable): "
          f"{(n_pos - eval_raw[eval_raw.type.isin(['TRANSFER','CASH_OUT'])].isFraud.sum())} of {n_pos}")

    # Plot PR curve with the chosen 0.90 point marked.
    chosen_i = int(np.argmin(np.abs(thr - DECISION_THRESHOLD)))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#2980b9", lw=1.5, label=f"PR-AUC={chosen['PR_AUC']:.3f}")
    ax.scatter([rec[chosen_i]], [prec[chosen_i]], color="#c0392b", zorder=5,
               label=f"chosen thr={DECISION_THRESHOLD:.2f}")
    ax.axhline(y_true.mean(), color="grey", ls="--", lw=0.8,
               label=f"no-skill={y_true.mean():.4f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall curve (XGBoost, eval split)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = repo_root() / "reports" / "pr_curve_eval.png"
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
