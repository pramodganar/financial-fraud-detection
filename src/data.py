"""Data access for the fraud project.

Reads from the CSV today. The brief mentions the data also lives in a
`fraud_detection` table inside Classification.db, so the loader is written so
swapping to sqlite is a one-line change (set source="sqlite").
"""

from pathlib import Path
import sqlite3

import pandas as pd

# Column dtypes: step is an integer hour index, type is categorical, the name
# columns are strings, balances/amount are floats, the two flags are 0/1 ints.
# Applied on both the CSV and sqlite paths so the sources yield the same frame.
DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def repo_root() -> Path:
    """Project root from this file's location, so paths don't depend on cwd."""
    return Path(__file__).resolve().parents[1]


def csv_path() -> Path:
    return repo_root() / "data" / "raw" / "fraud_detection.csv"


def load_data(source: str = "csv", nrows: int | None = None) -> pd.DataFrame:
    """Load the raw transactions.

    source: "csv" (default) reads data/raw/fraud_detection.csv.
            "sqlite" reads the fraud_detection table from Classification.db.
    nrows:  optional row cap, handy for quick iteration on the 6.3M-row file.
    """
    if source == "csv":
        df = pd.read_csv(csv_path(), dtype=DTYPES, nrows=nrows)
    elif source == "sqlite":
        db = repo_root() / "data" / "Classification.db"
        # ORDER BY step so the positional time-ordered split holds regardless of
        # table insertion order; LIMIT applies after ordering. Parameterized.
        query = "SELECT * FROM fraud_detection ORDER BY step"
        params: list = []
        if nrows is not None:
            query += " LIMIT ?"
            params = [nrows]
        with sqlite3.connect(db) as conn:
            df = pd.read_sql(query, conn, params=params)
        # read_sql infers dtypes; align with the CSV path so downstream code
        # sees identical frames regardless of source.
        df = df.astype({c: t for c, t in DTYPES.items() if c in df.columns})
    else:
        raise ValueError(f"unknown source: {source!r} (use 'csv' or 'sqlite')")

    # The train/eval/prod split is positional and assumes time order; guard it.
    if "step" in df.columns:
        assert df["step"].is_monotonic_increasing, "rows not time-ordered by step"
    return df


if __name__ == "__main__":
    # Smoke test: load a small slice and show shape + class balance.
    df = load_data(nrows=100_000)
    print(df.shape)
    print(df["isFraud"].value_counts())
