"""Describe a prepared dataset with a few numbers that may explain when
personalisation (and trajectory signals) can work.

    TRAJRISK_CONFIG=configs/movielens.yaml PYTHONPATH=src python scripts/dataset_profile.py

Run scripts/run_baselines.py first: the personalisation headroom is read from its results.
Saves <output_dir>/profile.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trajrisk import diffrec_data as dd
from trajrisk.config import load_config

PERSONAL = ["co-occurrence", "conditional", "itemknn"]
GROUPS = ["all", "1", "2-4", "5-19", "20-99", "100+"]


def gini(x):
    """0 = every item equally popular, 1 = one item gets everything."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    cfg = load_config()
    items, inter, cases = dd.load_processed(cfg)
    n_items = len(items)
    train, _ = dd.train_matrix(inter, n_items)

    item_counts = np.sort(np.asarray(train.sum(axis=0)).ravel())[::-1]
    per_user = np.asarray(train.sum(axis=1)).ravel()
    vm = cases[cases["split"] == "valid_model"]
    cases_per_user = vm.groupby("user_id").size().sort_values(ascending=False)

    profile = {
        "dataset": cfg["data"]["dataset"],
        "train_users": int(train.shape[0]),
        "items": n_items,
        "train_positives": int(train.nnz),
        "density_percent": round(100 * train.nnz / (train.shape[0] * n_items), 4),
        "median_positives_per_user": float(np.median(per_user)),
        "share_top1pct_items": round(float(item_counts[:max(1, n_items // 100)].sum()
                                           / item_counts.sum()), 3),
        "item_popularity_gini": round(gini(item_counts), 3),
        "valid_cases": int(len(vm)),
        "median_history_per_case": float(vm["n_prior"].median()),
        "share_cases_top1pct_users": round(float(
            cases_per_user.iloc[:max(1, len(cases_per_user) // 100)].sum() / len(vm)), 3),
    }

    bl_path = Path(cfg["paths"]["output_dir"]) / "baselines" / "valid_model.csv"
    if bl_path.exists():
        bl = pd.read_csv(bl_path, dtype={"group": str})
        r = bl.pivot(index="group", columns="method", values="recall@10")
        headroom = r[PERSONAL].max(axis=1) / r["popularity"] - 1
        profile["personalisation_headroom"] = {
            g: round(float(headroom[g]), 3) for g in GROUPS if g in headroom.index}
        profile["best_personal_method"] = r.loc["all", PERSONAL].idxmax()
    else:
        print("No baseline results yet: run scripts/run_baselines.py first.")

    out = Path(cfg["paths"]["output_dir"]) / "profile.json"
    with open(out, "w") as f:
        json.dump(profile, f, indent=2)
    print(json.dumps(profile, indent=2))
    print("Saved to", out)


if __name__ == "__main__":
    main()