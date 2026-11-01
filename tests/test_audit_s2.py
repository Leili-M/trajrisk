"""Proposal section 2: unit tests for the dataset audit (scripts/audit_data_s2.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "audit_data_s2", Path(__file__).resolve().parents[1] / "scripts" / "audit_data_s2.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _df(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "date"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def test_duplicates_are_counted():
    df = _df([(1, 10, 5, "2020-01-01"), (1, 10, 5, "2020-01-01"),   # identical row
              (1, 10, 3, "2020-02-01"),                            # same pair, later
              (2, 11, 4, "2020-01-01")])
    d = audit.duplicates(df)
    assert d["identical_rows"] == 1
    assert d["same_user_item_same_time"] == 1
    assert d["user_item_pairs_seen_more_than_once"] == 1
    assert d["extra_rows_from_repeated_pairs"] == 2


def test_eligible_needs_a_strictly_earlier_positive():
    df = _df([(1, 10, 5, "2020-01-01"), (1, 11, 5, "2020-03-01"),   # eligible
              (2, 12, 5, "2020-03-01"), (2, 13, 5, "2020-03-01"),   # same day only: not eligible
              (3, 14, 2, "2020-01-01"), (3, 15, 5, "2020-03-01")])  # earlier event not positive
    pos = df["rating"] >= 4
    e = audit.eligible_after(df, pos, pd.Timestamp("2020-02-01"))
    assert e == {"events": 1, "users": 1}


def test_nonempty():
    assert not any(audit.nonempty(v) for v in [None, np.nan, "", " ", "None", "[]", [], {}])
    assert all(audit.nonempty(v) for v in ["x", ["a"], {"k": 1}, 0, 3.5])
