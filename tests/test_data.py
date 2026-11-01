import pandas as pd

from trajrisk.data import add_split

CFG = {"data": {"split": {"train_end": "2010-01-10",
                          "valid_model_end": "2010-01-20",
                          "valid_end": "2010-01-30"}}}


def _df(dates):
    n = len(dates)
    return pd.DataFrame({"user_id": range(n), "item_id": range(n),
                         "rating": [5.0] * n, "date": pd.to_datetime(dates)})


def test_boundary_day_belongs_to_earlier_part():
    df = _df(["2010-01-09", "2010-01-10", "2010-01-11", "2010-01-20",
              "2010-01-21", "2010-01-30", "2010-01-31"])
    assert add_split(df, CFG)["split"].tolist() == [
        "train", "train", "valid_model", "valid_model",
        "valid_risk", "valid_risk", "test"]


def test_add_split_does_not_change_input():
    df = _df(["2010-01-09", "2010-01-25"])
    before = df.copy()
    add_split(df, CFG)
    pd.testing.assert_frame_equal(df, before)


def test_boundaries_with_seconds():
    """MovieLens uses exact times: one second after the boundary is the next part."""
    cfg = {"data": {"split": {"train_end": "2000-11-22 03:06:26",
                              "valid_model_end": "2000-11-26 06:42:19",
                              "valid_end": "2000-12-02 14:52:18"}}}
    df = _df(["2000-11-22 03:06:26", "2000-11-22 03:06:27"])
    assert add_split(df, cfg)["split"].tolist() == ["train", "valid_model"]
