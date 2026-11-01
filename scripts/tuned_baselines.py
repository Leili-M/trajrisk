"""Proposal section 4: tuned conventional baselines next to DiffRec, on the fixed split.

1. Tune every baseline with settings on valid_model (by NDCG@10, the same criterion
   used to select the DiffRec epoch):
     recent popularity: window of 30 / 90 / 180 / 365 days
     ItemKNN:           neighbourhood size k and shrink term
     EASE:              L2 penalty lambda
   Methods without settings (popularity, co-occurrence, conditional) are run as they are.
2. Evaluate the chosen setting of every method once on the test split, together
   with the frozen DiffRec models (one per seed).

    sbatch --partition=gpumedium --time=03:00:00 jobs/run.sh configs/amazon_video_games.yaml \
        scripts/tuned_baselines.py --config configs/amazon_video_games.yaml \
        --diffrec-runs vg_base vg_seed1 vg_seed2 vg_seed3 vg_seed4

Writes reports/baselines/<dataset>.md and .json in the repository and the
per-setting validation results to <output_dir>/baselines/tuning_valid_model.csv.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajrisk import baselines as bl
from trajrisk import diffrec_data as dd
from trajrisk import evaluate as ev

REPO = Path(__file__).resolve().parents[1]
GRID = {
    "recent popularity": [{"days": d} for d in (30, 90, 180, 365)],
    "itemknn": [{"k": k, "shrink": s} for k in (20, 50, 100, 200, 500, None) for s in (0, 10, 100)],
    "ease": [{"lam": lam} for lam in (50, 200, 500, 1000, 3000, 10000)],
}
FIXED = ["popularity", "co-occurrence", "conditional"]
KS = [10, 50]


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


class Cases:
    """History, seen items and targets of one split, built once."""

    def __init__(self, cases, inter, n_items, split):
        part = cases[cases["split"] == split].reset_index(drop=True)
        self.hist, self.seen = dd.case_matrices(part, inter, n_items)
        self.targets = part["target_idx"].to_numpy()
        self.n = len(part)

    def evaluate(self, fn, device):
        t0 = time.time()
        ranks = ev.rank_cases(fn, self.hist, self.seen, self.targets, device)
        return {**ev.metrics_from_ranks(ranks, KS), "seconds": round(time.time() - t0, 1)}


def build(name, setting, ctx):
    if name == "popularity":
        return bl.popularity(ctx["train"], ctx["device"])
    if name == "co-occurrence":
        return bl.co_occurrence(ctx["train"], ctx["device"])
    if name == "conditional":
        return bl.conditional(ctx["train"], ctx["device"])
    if name == "recent popularity":
        return bl.recent_popularity(ctx["inter"], ctx["n_items"], setting["days"], ctx["device"])
    if name == "itemknn":
        return bl.item_knn_pruned(ctx["G"], ctx["counts"], setting["k"], setting["shrink"], ctx["device"])
    if name == "ease":
        return bl.ease(ctx["G"], setting["lam"], ctx["device"])
    raise ValueError(name)


def diffrec_fn(cfg, run, n_items, device):
    from trajrisk.diffrec_model import build_diffrec, recommend_scores
    run_dir = Path(cfg["paths"]["output_dir"]) / "models" / run
    with open(run_dir / "config.json") as f:
        cfg = dict(cfg, model=json.load(f)["model"])
    model, diffusion = build_diffrec(cfg, n_items, device)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return lambda h: recommend_scores(model, diffusion, h)


def to_markdown(name, tuning, test, chosen, n_valid, n_test):
    L = [f"# Proposal section 4: tuned baselines and DiffRec on {name}", "",
         f"Settings chosen on valid_model ({n_valid:,} cases) by NDCG@10; "
         f"each chosen method was then run once on test ({n_test:,} cases).", "",
         "## Test results", "", "| Method | Setting | Recall@10 | NDCG@10 | Recall@50 |", "|---|---|---|---|---|"]
    for m, r in test.items():
        L.append(f"| {m} | {r.get('setting', '')} | {r['recall@10']:.4f} | {r['ndcg@10']:.4f} | {r['recall@50']:.4f} |")
    L += ["", "## Tuning on valid_model (NDCG@10)", "", "| Method | Setting | NDCG@10 | Recall@10 |", "|---|---|---|---|"]
    for r in tuning:
        star = " **(chosen)**" if chosen.get(r["method"]) == r["setting"] else ""
        L.append(f"| {r['method']} | {r['setting']}{star} | {r['ndcg@10']:.4f} | {r['recall@10']:.4f} |")
    return "\n".join(L) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--diffrec-runs", nargs="*", default=[])
    parser.add_argument("--only", nargs="*", help="limit to these methods (quick test)")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.config).stem

    items, inter, cases = dd.load_processed(cfg)
    n_items = len(items)
    train, _ = dd.train_matrix(inter, n_items)
    ctx = {"train": train, "inter": inter, "n_items": n_items, "device": device,
           "counts": bl.item_counts(train)}
    methods = [m for m in FIXED + list(GRID) if not args.only or m in args.only]
    if {"itemknn", "ease"} & set(methods):
        t0 = time.time()
        ctx["G"] = bl.gram_matrix(train)
        print(f"item x item counts: {n_items:,} items in {time.time() - t0:.0f} s", flush=True)

    valid = Cases(cases, inter, n_items, "valid_model")
    tuning, chosen = [], {}
    for m in methods:
        best = None
        for setting in GRID.get(m, [{}]):
            r = valid.evaluate(build(m, setting, ctx), device)
            row = {"method": m, "setting": json.dumps(setting), **r}
            tuning.append(row)
            print(f"  valid {m:18s} {row['setting']:28s} ndcg@10={r['ndcg@10']:.4f} "
                  f"recall@10={r['recall@10']:.4f} ({r['seconds']} s)", flush=True)
            if best is None or r["ndcg@10"] > best["ndcg@10"]:
                best = row
            torch.cuda.empty_cache()
        chosen[m] = best["setting"]

    test = Cases(cases, inter, n_items, "test")
    results = {}
    for m in methods:
        r = test.evaluate(build(m, json.loads(chosen[m]), ctx), device)
        results[m] = {"setting": chosen[m] if m in GRID else "", **r}
        print(f"  test  {m:18s} ndcg@10={r['ndcg@10']:.4f} recall@10={r['recall@10']:.4f}", flush=True)
        torch.cuda.empty_cache()
    seeds = []
    for run in args.diffrec_runs:
        r = test.evaluate(diffrec_fn(cfg, run, n_items, device), device)
        seeds.append(r)
        print(f"  test  DiffRec {run:12s} ndcg@10={r['ndcg@10']:.4f} recall@10={r['recall@10']:.4f}", flush=True)
    if seeds:
        df = pd.DataFrame(seeds)
        results["DiffRec"] = {"setting": f"mean of {len(seeds)} seeds",
                              **{c: float(df[c].mean()) for c in df.columns if c != "seconds"},
                              "sd_ndcg@10": float(df["ndcg@10"].std(ddof=1)) if len(seeds) > 1 else 0.0,
                              "per_seed": seeds}

    out_dir = Path(cfg["paths"]["output_dir"]) / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tuning).to_csv(out_dir / "tuning_valid_model.csv", index=False)
    rep = REPO / "reports" / "baselines"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"{name}.json").write_text(json.dumps({"chosen": chosen, "tuning": tuning, "test": results}, indent=2))
    (rep / f"{name}.md").write_text(to_markdown(name, tuning, results, chosen, valid.n, test.n))
    print(to_markdown(name, tuning, results, chosen, valid.n, test.n))


if __name__ == "__main__":
    main()
