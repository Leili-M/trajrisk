import pandas as pd

from trajrisk import diffrec_data as dd
from trajrisk.truncate import last_m_history


def _data():
    inter = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 2],
        "item_idx": [0, 1, 2, 3, 1],
        "rating": [5.0, 5.0, 2.0, 5.0, 4.0],
        "date": pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-05", "2000-01-01"]),
        "split": ["train"] * 5,
        "positive": [True, True, False, True, True],
    })
    cases = pd.DataFrame({
        "case_id": [0, 1], "user_id": [1, 2],
        "date": pd.to_datetime(["2000-01-04", "2000-01-02"]),
        "split": ["valid_model"] * 2, "target_idx": [3, 2], "n_prior": [2, 1],
    })
    return inter, cases


def test_large_m_gives_the_full_history():
    inter, cases = _data()
    full, _ = dd.case_matrices(cases, inter, 4)
    assert (last_m_history(cases, inter, 4, 100) != full).nnz == 0


def test_keeps_only_the_most_recent_liked_items():
    inter, cases = _data()
    h = last_m_history(cases, inter, 4, 1).toarray()
    assert h[0].tolist() == [0, 1, 0, 0]   # user 1 liked items 0 and 1 before; 1 is the latest
    assert h[1].tolist() == [0, 1, 0, 0]




def test_random_keeps_m_items_from_the_full_history():

    from trajrisk.truncate import random_m_history

    inter, cases = _data()

    full, _ = dd.case_matrices(cases, inter, 4)

    h = random_m_history(cases, inter, 4, 1, seed=0)

    assert h.sum(axis=1).A1.tolist() == [1, 1]

    assert (h.multiply(full) != h).nnz == 0

