"""Feature importance for the selected XGBoost model.

Permutation importance is the headline (model-agnostic, measured as the drop in
PR-AUC when a column is shuffled). XGBoost's native gain is shown alongside as a
cheap cross-check. Computed on a sample of the eval split for speed.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from src.data import load_data, repo_root
from src.model_selection import build_models, TRAIN_END, EVAL_END
from src.pipeline import make_features, FEATURE_COLUMNS

SAMPLE = 200_000  # eval rows used for the permutation pass
SEED = 0


def main():
    df = load_data(nrows=EVAL_END)
    X_tr, y_tr = make_features(df.iloc[:TRAIN_END])
    X_ev, y_ev = make_features(df.iloc[TRAIN_END:EVAL_END])

    model = build_models(y_tr)["xgboost"]
    model.fit(X_tr, y_tr)

    # Sample the eval features (keep it reproducible).
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_ev), size=min(SAMPLE, len(X_ev)), replace=False)
    Xs, ys = X_ev.iloc[idx], y_ev.iloc[idx]
    print(f"permutation sample: {len(Xs)} rows, {int(ys.sum())} positives")

    perm = permutation_importance(
        model, Xs, ys, scoring="average_precision",
        n_repeats=5, random_state=SEED, n_jobs=1,
    )
    order = np.argsort(perm.importances_mean)[::-1]
    print("\npermutation importance (mean PR-AUC drop):")
    for i in order:
        print(f"  {FEATURE_COLUMNS[i]:18s} {perm.importances_mean[i]:.4f} "
              f"+/- {perm.importances_std[i]:.4f}")

    # Native gain (the xgb step is the last in the pipeline).
    xgb = model.steps[-1][1]
    gain = xgb.feature_importances_

    # Two-panel plot.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = np.array(FEATURE_COLUMNS)

    o = np.argsort(perm.importances_mean)
    axes[0].barh(names[o], perm.importances_mean[o], color="#2980b9")
    axes[0].set_title("Permutation importance (PR-AUC drop)")

    o2 = np.argsort(gain)
    axes[1].barh(names[o2], gain[o2], color="#27ae60")
    axes[1].set_title("XGBoost native gain")

    fig.tight_layout()
    out = repo_root() / "reports" / "feature_importance.png"
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
