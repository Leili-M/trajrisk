"""Stage 2: build the frozen dataset that every later stage reads."""

import json
from pathlib import Path

import pandas as pd

from .data import add_split, load_interactions

EVAL_SPLITS = ["valid_model", "valid_risk", "test"]


def build_dataset(cfg):
    df = add_split(load_interactions(cfg), cfg)
    positive = df["rating"].isin(cfg["data"]["positive_ratings"])

    # 1. Recipe set: recipes with at least k positives in training
    k = cfg["data"]["min_item_train_positives"]
    train_counts = df.loc[positive & (df["split"] == "train"), "item_id"].value_counts()
    kept_ids = sorted(train_counts[train_counts >= k].index)
    items = pd.DataFrame({"item_id": kept_ids, "item_idx": range(len(kept_ids))})

    # 2. Keep only interactions on kept recipes and add the model index
    inter = df.merge(items, on="item_id")
    inter["positive"] = inter["rating"].isin(cfg["data"]["positive_ratings"])
    inter = inter[["user_id", "item_idx", "rating", "date", "split", "positive"]]

    # 3. History size: positives on kept recipes on STRICTLY earlier days
    pos = inter[inter["positive"]]
    daily = pos.groupby(["user_id", "date"]).size().rename("n").reset_index()
    daily["n_prior"] = daily.groupby("user_id")["n"].cumsum() - daily["n"]
    pos = pos.merge(daily[["user_id", "date", "n_prior"]], on=["user_id", "date"])

    # 4. Evaluation cases: positive, outside training, with earlier history
    cases = pos[(pos["split"] != "train") & (pos["n_prior"] >= 1)]
    cases = (cases[["user_id", "date", "split", "item_idx", "n_prior"]]
             .rename(columns={"item_idx": "target_idx"})
             .sort_values(["split", "user_id", "date", "target_idx"])
             .reset_index(drop=True))
    cases.insert(0, "case_id", range(len(cases)))

    # 5. Attrition report: how many were lost at each condition
    report = {
        "settings": cfg["data"],
        "recipes_kept": len(items),
        "train_users": int(pos.loc[pos["split"] == "train", "user_id"].nunique()),
        "train_positives": int((pos["split"] == "train").sum()),
        "attrition": {},
    }
    for s in EVAL_SPLITS:
        report["attrition"][s] = {
            "interactions": int((df["split"] == s).sum()),
            "positive": int((positive & (df["split"] == s)).sum()),
            "positive_on_kept_recipes": int((pos["split"] == s).sum()),
            "cases_with_earlier_history": int((cases["split"] == s).sum()),
            "users": int(cases.loc[cases["split"] == s, "user_id"].nunique()),
        }
    return items, inter, cases, report


def save_dataset(cfg, items, inter, cases, report):
    out = Path(cfg["paths"]["data_dir"]) / "processed"
    out.mkdir(parents=True, exist_ok=True)
    items.to_parquet(out / "items.parquet", index=False)
    inter.to_parquet(out / "interactions.parquet", index=False)
    cases.to_parquet(out / "cases.parquet", index=False)
    with open(out / "prepare_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return out
