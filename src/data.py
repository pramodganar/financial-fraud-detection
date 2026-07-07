"""Data access for the fraud project.

Reads from the CSV today. The brief mentions the data also lives in a
`fraud_detection` table inside Classification.db, so the loader is written so
swapping to sqlite is a one-line change (set source="sqlite").
"""

from pathlib import Path
import sqlite3

import pandas as pd

# Column dtypes. step/balances are floats in the file; the name columns and
# type are strings; the two flags are 0/1 ints.
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
        return pd.read_csv(csv_path(), dtype=DTYPES, nrows=nrows)

    if source == "sqlite":
        db = repo_root() / "data" / "Classification.db"
        query = "SELECT * FROM fraud_detection"
        if nrows is not None:
            query += f" LIMIT {nrows}"
        with sqlite3.connect(db) as conn:
            return pd.read_sql(query, conn)

    raise ValueError(f"unknown source: {source!r} (use 'csv' or 'sqlite')")


if __name__ == "__main__":
    # Smoke test: load a small slice and show shape + class balance.
    df = load_data(nrows=100_000)
    print(df.shape)
    print(df["isFraud"].value_counts())
