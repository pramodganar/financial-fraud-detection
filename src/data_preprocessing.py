import pandas as pd
import numpy as np

# Transactions above this amount are flagged as a fraud indicator.
# Based on domain knowledge: the dataset's fraud cases are concentrated in large transfers.
LARGE_TRANSACTION_THRESHOLD = 200_000


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features from raw transaction data.

    Steps:
    - Drops identifier columns (nameOrig, nameDest) — high cardinality, no signal.
    - One-hot encodes transaction type (drop_first=True to avoid multicollinearity).
    - Creates three balance-difference features that are strong fraud signals.
    - Replaces inf values and fills NaN with 0.

    NaN fill rationale: NaN in newbalanceOrig occurs only for CASH_IN transactions
    where the origin account balance is not tracked by the payment system.
    Filling with 0 is semantically correct (not a distributional guess).
    Median imputation on the full dataset would leak test-set statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction DataFrame matching the PaySim dataset schema.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for train/test split and modeling.
    """
    df = df.copy()

    df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True, errors='ignore')

    # Manually one-hot encode 'type' instead of using pd.get_dummies().
    # pd.get_dummies(drop_first=True) at inference time drops the only dummy
    # column it creates when a single-row DataFrame contains just one category,
    # leaving all type columns as zero — the model sees no transaction type signal.
    # The reference category (dropped) is CASH_IN, matching training behaviour.
    for cat in ('CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'):
        df[f'type_{cat}'] = (df['type'] == cat).astype(int)
    df.drop('type', axis=1, inplace=True)

    df['orig_balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['dest_balance_diff'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['large_transaction'] = (df['amount'] > LARGE_TRANSACTION_THRESHOLD).astype(int)

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df
