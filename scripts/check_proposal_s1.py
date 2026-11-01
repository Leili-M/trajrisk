"""Proposal section 1 checks on the real processed data, for every dataset.

R1.1  each case history holds only positive interactions strictly before the case time
      (checked by recomputing it independently of case_matrices)
R1.2  eligible catalog I(tau): every target and every recommendable item has at least k
      positive training interactions, so every recommendable item already existed
      before any evaluation case; attrition per split shows what this rule removes
R1.4  the target is never among the items hidden as 'already seen'
Also: interactions and cases sit inside their frozen split boundaries.

    sbatch --time=01:00:00 jobs/run.sh configs/movielens.yaml scripts/check_proposal_s1.py \
        --configs configs/movielens.yaml configs/base.yaml \
                  configs/amazon_video_games.yaml configs/amazon_books.yaml

Writes reports/checks/proposal_s1.md and reports/checks/proposal_s1.json in the repository.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from trajrisk import diffrec_data as dd

REPO = Path(__file__).resolve().parents[1]
SPLITS = ["valid_model", "valid_risk", "test"]


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


def boundary_checks(inter, cases, split):
    te, vm, ve = (pd.Timestamp(split[k]) for k in ("train_end", "valid_model_end", "valid_end"))
    rng = {"train": (pd.Timestamp.min, te), "valid_model": (te, vm),
           "valid_risk": (vm, ve), "test": (ve, pd.Timestamp.max)}
    out = {}
    for name, frame in (("interactions", inter), ("cases", cases)):
        ok = True
        for s, (lo, hi) in rng.items():
            d = frame.loc[frame["split"] == s, "date"]
            if len(d):
                ok &= bool(((d > lo) | (s == "train")).all() and (d <= hi).all())
        out[f"{name}_inside_split_boundaries"] = ok
    out["no_case_in_training_period"] = bool((cases["split"] != "train").all())
    return out


def catalog_checks(items, inter, cases, k):
    pos_train = inter[(inter["split"] == "train") & inter["positive"]]
    counts = pos_train["item_idx"].value_counts().reindex(range(len(items)), fill_value=0)
    return {
        "catalog_items": int(len(items)),
        "min_positive_training_interactions_per_item": int(counts.min()),
        "every_item_has_k_training_positives": bool((counts >= k).all()),
        "every_target_in_catalog": bool(cases["target_idx"].between(0, len(items) - 1).all()),
    }


def history_checks(items, inter, cases, n_sample, seed):
    samp = (cases[cases["split"].isin(SPLITS)]
            .groupby("split", group_keys=False)
            .apply(lambda g: g.sample(min(n_sample, len(g)), random_state=seed))
            .reset_index(drop=True))
    hist, seen = dd.case_matrices(samp, inter, len(items))

    # Independent recomputation with a plain merge
    m = samp[["case_id", "user_id", "date"]].merge(
        inter[["user_id", "item_idx", "date", "positive"]], on="user_id", suffixes=("", "_event"))
    earlier_pos = m[(m["date_event"] < m["date"]) & m["positive"]]
    order = samp["case_id"]
    n_events = earlier_pos.groupby("case_id").size().reindex(order, fill_value=0).to_numpy()
    n_unique = earlier_pos.groupby("case_id")["item_idx"].nunique().reindex(order, fill_value=0).to_numpy()

    row_of = dict(zip(samp["case_id"], range(len(samp))))
    allowed = set(zip(earlier_pos["case_id"].map(row_of), earlier_pos["item_idx"]))
    r, c = hist.nonzero()
    in_history = set(zip(r.tolist(), c.tolist()))

    rows = np.arange(len(samp))
    tgt = samp["target_idx"].to_numpy()
    return {
        "cases_checked": int(len(samp)),
        "history_items_all_strictly_earlier": in_history <= allowed,
        "history_size_matches_recomputation": bool((np.asarray(hist.sum(1)).ravel() == n_unique).all()),
        "n_prior_matches_earlier_positive_events": bool((n_events == samp["n_prior"].to_numpy()).all()),
        "every_case_has_history": bool((samp["n_prior"] >= 1).all()),
        "target_never_hidden_as_seen": bool((np.asarray(seen[rows, tgt]).ravel() == 0).all()),
        "target_never_in_history": bool((np.asarray(hist[rows, tgt]).ravel() == 0).all()),
    }


def attrition(cfg):
    path = Path(cfg["paths"]["data_dir"]) / "processed" / "prepare_report.json"
    with open(path) as f:
        rep = json.load(f)
    return rep["attrition"], rep.get("recipes_kept"), rep.get("train_users")


def to_markdown(results):
    lines = ["# Proposal section 1: checks on the processed data", ""]
    for name, r in results.items():
        lines += [f"## {name}", "", f"Catalog rule: items with at least {r['k']} positive "
                  f"training interactions ({r['catalog']['catalog_items']:,} items).", "",
                  "| Check | Result |", "|---|---|"]
        for group in ("boundaries", "catalog", "history"):
            for key, val in r[group].items():
                shown = ("PASS" if val else "FAIL") if isinstance(val, bool) else f"{val:,}"
                lines.append(f"| {key.replace('_', ' ')} | {shown} |")
        lines += ["", "| Split | Interactions | Positive | Positive on catalog items | "
                  "Cases with earlier history | Users |", "|---|---|---|---|---|---|"]
        for s, a in r["attrition"].items():
            lines.append(f"| {s} | {a['interactions']:,} | {a['positive']:,} | "
                         f"{a['positive_on_kept_recipes']:,} | {a['cases_with_earlier_history']:,} | "
                         f"{a['users']:,} |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--sample", type=int, default=20000, help="cases checked per split")
    args = parser.parse_args()

    results, all_ok = {}, True
    for path in args.configs:
        cfg = load_cfg(path)
        name = Path(path).stem
        items, inter, cases = dd.load_processed(cfg)
        k = cfg["data"]["min_item_train_positives"]
        att, _, _ = attrition(cfg)
        r = {"k": k,
             "boundaries": boundary_checks(inter, cases, cfg["data"]["split"]),
             "catalog": catalog_checks(items, inter, cases, k),
             "history": history_checks(items, inter, cases, args.sample, cfg["project"]["seed"]),
             "attrition": att}
        ok = all(v for g in ("boundaries", "catalog", "history")
                 for v in r[g].values() if isinstance(v, bool))
        all_ok &= ok
        results[name] = r
        print(f"{name}: {'ALL PASS' if ok else 'SOME CHECKS FAIL'}")
        for g in ("boundaries", "catalog", "history"):
            for key, val in r[g].items():
                print(f"   {key:48s} {val}")

    out = REPO / "reports" / "checks"
    out.mkdir(parents=True, exist_ok=True)
    (out / "proposal_s1.json").write_text(json.dumps(results, indent=2, default=str))
    (out / "proposal_s1.md").write_text(to_markdown(results))
    print(f"\n{'ALL DATASETS PASS' if all_ok else 'SOME CHECKS FAIL'}  ->  {out / 'proposal_s1.md'}")


if __name__ == "__main__":
    main()
