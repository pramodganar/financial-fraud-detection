"""Cover the sqlite branch of load_data with a temp DB.

The project ships a fraud_detection table in Classification.db but I worked off the
CSV, so this branch was never exercised. Build a tiny DB and read it back so the
branch is tested rather than dead.
"""

import sqlite3

import pandas as pd
import pytest

from src import data


def _tiny_frame():
    return pd.DataFrame({
        "type": ["TRANSFER", "CASH_OUT"],
        "amount": [181.0, 200.0],
        "oldbalanceOrg": [181.0, 200.0],
        "newbalanceOrig": [0.0, 0.0],
        "oldbalanceDest": [0.0, 0.0],
        "newbalanceDest": [0.0, 200.0],
        "isFraud": [0, 1],
    })


def test_sqlite_loader_reads_table(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    db = tmp_path / "data" / "Classification.db"
    with sqlite3.connect(db) as conn:
        _tiny_frame().to_sql("fraud_detection", conn, index=False)

    monkeypatch.setattr(data, "repo_root", lambda: tmp_path)
    out = data.load_data(source="sqlite")
    assert {"type", "amount", "isFraud"}.issubset(out.columns)
    assert len(out) == 2


def test_sqlite_loader_honours_nrows(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    db = tmp_path / "data" / "Classification.db"
    with sqlite3.connect(db) as conn:
        _tiny_frame().to_sql("fraud_detection", conn, index=False)

    monkeypatch.setattr(data, "repo_root", lambda: tmp_path)
    out = data.load_data(source="sqlite", nrows=1)
    assert len(out) == 1


def test_load_data_rejects_unknown_source():
    with pytest.raises(ValueError):
        data.load_data(source="parquet")
