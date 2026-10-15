import pandas as pd

from trajrisk.diffrec_data import case_matrices


def test_history_uses_only_earlier_days():
    inter = pd.DataFrame({
        "user_id":  [1, 1, 1, 1],
        "item_idx": [0, 1, 2, 3],
        "date": pd.to_datetime(["2011-01-01", "2011-01-01",   # earlier day
                                "2011-01-05", "2011-01-05"]), # same day as the case
        "positive": [True, False, True, True],
    })
    cases = pd.DataFrame({"user_id": [1], "date": pd.to_datetime(["2011-01-05"])})

    history, seen = case_matrices(cases, inter, n_items=4)

    assert history.toarray().tolist() == [[1, 0, 0, 0]]   # only the earlier positive
    assert seen.toarray().tolist() == [[1, 1, 0, 0]]      # earlier day, any rating
