# EDA findings

Source: `notebooks/eda.ipynb`, run on the full 6,362,620-row file. Notebook
outputs are stripped for clean diffs; every number and plot below regenerates by
running the notebook top to bottom against `data/raw/fraud_detection.csv`.

## Class balance
- 8,213 fraud rows out of 6,362,620 → **0.129%** positive. Accuracy is useless;
  evaluation is on PR-AUC / precision / recall.

## Fraud only occurs in TRANSFER and CASH_OUT (confirmed)
| type     | n         | frauds | fraud rate |
|----------|-----------|--------|-----------|
| TRANSFER | 515,688   | 3,931  | 0.762%    |
| CASH_OUT | 2,165,868 | 3,991  | 0.184%    |
| CASH_IN  | 1,354,742 | 0      | 0%        |
| PAYMENT  | 2,082,597 | 0      | 0%        |
| DEBIT    | 40,122    | 0      | 0%        |

Every fraud row is TRANSFER or CASH_OUT. The other three types carry zero fraud
signal — the pipeline filters to these two types and drops the rest.

## isFlaggedFraud is near useless
Fires only **16 times** in 6.36M rows, all of them true fraud — but it misses
8,197 of 8,213 frauds. Keeping it as a documented curiosity, not a real feature.

## Amount
Fraud transactions are much larger: median fraud amount ≈ 441k vs ≈ 75k for
legit; mean ≈ 1.47M vs 178k. Strongly right-skewed, so a log transform is
appropriate (see `amount_distribution.png`).

## Balance-error signal (the important one)
Raw balances are inconsistent — they fail to reconcile a lot:
- `errorBalanceOrig != 0` for **76.5%** of all rows.
- `errorBalanceDest != 0` for **63.7%** of all rows.

Within TRANSFER/CASH_OUT the error separates the classes cleanly:
- mean `errorBalanceOrig`: legit ≈ 286,854 vs fraud ≈ 11,465 (fraud drains the
  origin account to zero, so the error nearly vanishes).
- mean `errorBalanceDest`: legit ≈ −30,897 vs fraud ≈ 725,737 (fraudulent
  destinations don't credit the money the way a real account would).

This is why we engineer `errorBalanceOrig`/`errorBalanceDest` instead of feeding
the raw balance columns. See `balance_error.png`.

## Zero-balance and merchant patterns
- Destination `oldbalanceDest == 0`: **65%** of frauds vs **14%** of legit
  (within T/CO) — a useful flag.
- Origin `newbalanceOrig == 0`: 94% fraud vs 86% legit (consistent with accounts
  being emptied).
- Merchant destinations (`nameDest` starts with `M`): **0** frauds out of
  2.15M rows. Merchants are never fraud targets here, so `isMerchantDest` is a
  clean negative signal.

## Time split — a real caveat
The prescribed time-ordered split is *not* class-stationary:

| split | rows      | frauds | rate    |
|-------|-----------|--------|---------|
| train (first 4M)  | 4,000,000 | 3,381 | 0.085% |
| eval  (next 1M)   | 1,000,000 |   554 | 0.055% |
| prod  (last ~1.36M)| 1,362,620 | 4,278 | 0.314% |

Fraud density rises sharply in the held-out production tail (see
`fraud_over_time.png`). We respect the split anyway because it mirrors real
deployment over time, but it means the production-set metrics will differ from
eval, and we should not be surprised by that. Worth one line in the final report.

## Plots
- `reports/fraud_rate_by_type.png`
- `reports/amount_distribution.png`
- `reports/balance_error.png`
- `reports/fraud_over_time.png`
