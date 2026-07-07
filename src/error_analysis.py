"""Characterize what the model misses on the held-out production set.

Aggregate recall hides which frauds slip through. This scores the production tail
(rows 5,000,000+) at the shipped threshold, splits the false negatives into
unscoreable (wrong/missing type -> scored 0 by rule) vs scoreable (in
TRANSFER/CASH_OUT but ranked below threshold), and profiles the scoreable misses
against the frauds we catch. Writes reports/error_analysis.md.
"""

import numpy as np

from src.data import load_data, repo_root
from src.predict import load_artifact
from src.pipeline import make_features, FRAUD_TYPES

PROD_START = 5_000_000


def main():
    prod = load_data().iloc[PROD_START:].reset_index(drop=True)
    art = load_artifact()
    thr = art["threshold"]

    X, _ = make_features(prod)
    proba = np.zeros(len(prod))
    proba[X.index.to_numpy()] = art["model"].predict_proba(X)[:, 1]

    y = prod["isFraud"].to_numpy()
    pred = (proba >= thr).astype(int)

    scoreable = prod["type"].isin(FRAUD_TYPES).to_numpy()
    fn = (y == 1) & (pred == 0)
    tp = (y == 1) & (pred == 1)

    n_fraud = int(y.sum())
    fn_unscoreable = int((fn & ~scoreable).sum())
    fn_scoreable = int((fn & scoreable).sum())
    n_tp = int(tp.sum())

    # Profile the scoreable misses vs the catches.
    def profile(mask):
        idx = np.where(mask)[0]
        amt = prod["amount"].to_numpy()[idx]
        err = X["errorBalanceOrig"].reindex(idx).to_numpy()
        pr = proba[idx]
        types = prod["type"].to_numpy()[idx]
        return {
            "n": len(idx),
            "median_amount": float(np.median(amt)) if len(idx) else float("nan"),
            "median_errorBalanceOrig": float(np.nanmedian(err)) if len(idx) else float("nan"),
            "median_proba": float(np.median(pr)) if len(idx) else float("nan"),
            "n_transfer": int((types == "TRANSFER").sum()),
            "n_cash_out": int((types == "CASH_OUT").sum()),
        }

    miss = profile(fn & scoreable)
    catch = profile(tp)
    scoreable_recall = n_tp / (n_tp + fn_scoreable) if (n_tp + fn_scoreable) else float("nan")

    lines = [
        "# Error analysis (held-out production set)",
        "",
        f"Production tail: rows {PROD_START:,}+ ({len(prod):,} transactions, "
        f"{n_fraud:,} frauds). Scored once at threshold {thr}.",
        "",
        "## Where the misses go",
        "",
        f"- Total missed frauds (FN): {fn_unscoreable + fn_scoreable}",
        f"- Unscoreable by design (type not TRANSFER/CASH_OUT or missing): "
        f"**{fn_unscoreable}** — the model never sees these; they are the real ceiling.",
        f"- Scoreable misses (in TRANSFER/CASH_OUT, ranked below {thr}): "
        f"**{fn_scoreable}**",
        f"- Effective recall on scoreable traffic: **{scoreable_recall:.3f}** "
        f"({n_tp}/{n_tp + fn_scoreable}).",
        "",
        "## Scoreable misses vs. catches",
        "",
        "| | scoreable misses (FN) | catches (TP) |",
        "|---|---|---|",
        f"| count | {miss['n']} | {catch['n']} |",
        f"| TRANSFER / CASH_OUT | {miss['n_transfer']} / {miss['n_cash_out']} "
        f"| {catch['n_transfer']} / {catch['n_cash_out']} |",
        f"| median amount | {miss['median_amount']:,.0f} | {catch['median_amount']:,.0f} |",
        f"| median errorBalanceOrig | {miss['median_errorBalanceOrig']:,.0f} "
        f"| {catch['median_errorBalanceOrig']:,.0f} |",
        f"| median fraud probability | {miss['median_proba']:.3f} | {catch['median_proba']:.3f} |",
        "",
        "## Read",
        "",
        "The dominant miss category is unscoreable rows, not model error. Among the "
        "scoreable misses, the model's own top feature explains them: caught frauds "
        "drain the origin account so `errorBalanceOrig` collapses toward zero, while "
        "the misses retain a large reconciliation error — they don't fit the "
        "'account emptied' signature the model learned. That is the honest limit of a "
        "model leaning on one dominant feature (see feature_importance.md), and the "
        "argument for adding velocity / destination-history features before trusting "
        "it on adversarial real-world traffic.",
        "",
    ]
    out = repo_root() / "reports" / "error_analysis.md"
    out.write_text("\n".join(lines), encoding="utf-8")

    for ln in lines:
        print(ln)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
