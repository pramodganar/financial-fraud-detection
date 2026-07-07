# Feature importance

Permutation importance (drop in PR-AUC when a column is shuffled) on a 200k-row
eval sample, with XGBoost native gain as a cross-check. Plot:
`reports/feature_importance.png`.

| feature | perm. importance (PR-AUC drop) |
|---------|-------------------------------|
| errorBalanceOrig | 0.934 |
| origZeroAfter    | 0.196 |
| oldbalanceOrg    | 0.025 |
| amount           | 0.007 |
| oldbalanceDest   | 0.005 |
| isTransfer       | 0.004 |
| errorBalanceDest | 0.004 |
| (remaining flags / log_amount / raw balances) | < 0.003 |

## Interpretation

The engineered balance features carry the model, exactly as the EDA predicted:

- **errorBalanceOrig dominates** (shuffling it costs ~0.93 PR-AUC — almost the
  entire signal). Fraud drains the origin account to zero, so its reconciliation
  error collapses to ~0; that single feature all but identifies fraud.
- **origZeroAfter** (origin balance ends at zero) is the clear second signal —
  the same "account emptied" pattern, captured as a flag.
- **oldbalanceOrg** and **amount** contribute modestly. Amount ranks lower than
  one might expect because errorBalanceOrig is itself a function of amount and the
  balances, so it already absorbs most of the amount signal.

A caveat on reading this: permutation importance *understates correlated
features*. errorBalanceDest separated the classes well in the EDA, but here it
scores low because errorBalanceOrig + origZeroAfter already capture the fraud
pattern, making it redundant rather than useless. The native-gain panel tells a
consistent story (origin-side balance features on top).

Bottom line: the balance-error engineering is what makes this model work. The raw
balance columns and the log-amount transform add little once the error features
are present, but they are cheap to keep and do no harm.
