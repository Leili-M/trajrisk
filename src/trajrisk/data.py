"""Load the raw data in the form the rest of the project uses.

Every dataset is turned into the same four columns:
user_id, item_id, rating, date.
"""

import zipfile
from pathlib import Path

import pandas as pd

HUMMUS_COLUMNS = {
    "new_member_id": "user_id",
    "new_recipe_id": "item_id",
    "rating": "rating",
    "last_modified_date": "date",
}


def _load_hummus(raw_dir):
    df = pd.read_csv(
        raw_dir / "pp_reviews.zip",
        usecols=list(HUMMUS_COLUMNS),
        dtype={"new_member_id": "int32", "new_recipe_id": "int32", "rating": "float32"},
    )
    df = df.rename(columns=HUMMUS_COLUMNS)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


def _load_movielens(raw_dir):
    with zipfile.ZipFile(raw_dir / "ml-1m.zip") as z:
        with z.open("ml-1m/ratings.dat") as f:
            df = pd.read_csv(f, sep="::", engine="python", header=None,
                             names=["user_id", "item_id", "rating", "timestamp"],
                             dtype={"user_id": "int32", "item_id": "int32",
                                    "rating": "float32", "timestamp": "int64"})
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    return df.drop(columns="timestamp")


LOADERS = {"hummus": _load_hummus, "movielens": _load_movielens}


def load_interactions(cfg):
    """Read interactions, drop non-review rows, and sort by user and date."""
    raw_dir = Path(cfg["paths"]["data_dir"]) / "raw"
    df = LOADERS[cfg["data"]["dataset"]](raw_dir)
    df = df[~df["rating"].isin(cfg["data"]["drop_ratings"])]
    df = df[["user_id", "item_id", "rating", "date"]]
    return df.sort_values(["user_id", "date"]).reset_index(drop=True)


def add_split(df, cfg):
    """Label each interaction as train, valid_model, valid_risk or test.

    Boundaries come from the frozen dates in the config. Each boundary
    belongs to the earlier part.
    """
    split = cfg["data"]["split"]
    train_end = pd.Timestamp(split["train_end"])
    valid_model_end = pd.Timestamp(split["valid_model_end"])
    valid_end = pd.Timestamp(split["valid_end"])

    df = df.copy()
    df["split"] = "test"
    df.loc[df["date"] <= valid_end, "split"] = "valid_risk"
    df.loc[df["date"] <= valid_model_end, "split"] = "valid_model"
    df.loc[df["date"] <= train_end, "split"] = "train"
    return df
