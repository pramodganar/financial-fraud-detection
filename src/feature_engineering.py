import pandas as pd


def create_features(df):

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

    return df