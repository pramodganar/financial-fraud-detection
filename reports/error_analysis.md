# Error analysis (held-out production set)

Production tail: rows 5,000,000+ (1,362,620 transactions, 4,278 frauds). Scored once at threshold 0.9.

## Where the misses go

- Total missed frauds (FN): 186
- Unscoreable by design (type not TRANSFER/CASH_OUT or missing): **148** — the model never sees these; they are the real ceiling.
- Scoreable misses (in TRANSFER/CASH_OUT, ranked below 0.9): **38**
- Effective recall on scoreable traffic: **0.991** (4092/4130).

## Scoreable misses vs. catches

| | scoreable misses (FN) | catches (TP) |
|---|---|---|
| count | 38 | 4092 |
| TRANSFER / CASH_OUT | 1 / 37 | 2056 / 2036 |
| median amount | 148,697 | 468,619 |
| median errorBalanceOrig | 399,045 | 0 |
| median fraud probability | 0.008 | 1.000 |

## Read

The dominant miss category is unscoreable rows, not model error. Among the scoreable misses, the model's own top feature explains them: caught frauds drain the origin account so `errorBalanceOrig` collapses toward zero, while the misses retain a large reconciliation error — they don't fit the 'account emptied' signature the model learned. That is the honest limit of a model leaning on one dominant feature (see feature_importance.md), and the argument for adding velocity / destination-history features before trusting it on adversarial real-world traffic.
