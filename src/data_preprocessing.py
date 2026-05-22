import pandas as pd
import numpy as np


def preprocess_data(df):

    # Drop irrelevant columns
    df.drop(
        ['nameOrig', 'nameDest'],
        axis=1,
        inplace=True,
        errors='ignore'
    )

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=['type'],
        drop_first=True
    )

    # Feature engineering
    df['orig_balance_diff'] = (
        df['oldbalanceOrg']
        - df['newbalanceOrig']
    )

    df['dest_balance_diff'] = (
        df['newbalanceDest']
        - df['oldbalanceDest']
    )

    df['large_transaction'] = (
        df['amount'] > 200000
    ).astype(int)

    # Handle infinite values
    df.replace(
        [np.inf, -np.inf],
        0,
        inplace=True
    )

    # Fill missing values
    df = df.fillna(df.median())

    return df