"""Proposal section 4: DiffRec and GiffCF side by side on the fixed split.

Every listed model run is scored once on valid_model and on test with the same
cases, seen-item masking and metrics as the tuned baselines
(reports/baselines/<dataset>.md), so the numbers can be put in one table.

    sbatch --partition=gpumedium --time=01:00:00 jobs/run.sh configs/amazon_video_games.yaml \
        scripts/eval_backbone.py --config configs/amazon_video_games.yaml \
        --runs vg_base vg_seed1 vg_seed2 vg_giff_s1 vg_giff_s2 vg_giff_s3

Writes reports/backbones/<dataset>.md and .json in the repository.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch

from trajrisk import diffrec_data as dd
from trajrisk import evaluate as ev
from trajrisk import giffcf as gc

REPO = Path(__file__).resolve().parents[1]
KS = [10, 50]


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


def score_fn(cfg, run, train, n_items, device):
    """(backbone name, function history -> final scores) for a saved run."""
    run_dir = Path(cfg["paths"]["output_dir"]) / "models" / run
    meta = json.loads((run_dir / "config.json").read_text())
    if meta.get("backbone") == "giffcf":
        model, _ = gc.load_run(run_dir, train, device)
        return "GiffCF", model.recommend
    from trajrisk.diffrec_model import build_diffrec, recommend_scores
    model, diffusion = build_diffrec(dict(cfg, model=meta["model"]), n_items, device)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return "DiffRec", lambda h: recommend_scores(model, diffusion, h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.config).stem
    items, inter, cases = dd.load_processed(cfg)
    n_items = len(items)
    train, _ = dd.train_matrix(inter, n_items)
    splits = {}
    for split in ("valid_model", "test"):
        part = cases[cases["split"] == split].reset_index(drop=True)
        hist, seen = dd.case_matrices(part, inter, n_items)
        splits[split] = (hist, seen, part["target_idx"].to_numpy())

    rows = []
    for run in args.runs:
        backbone, fn = score_fn(cfg, run, train, n_items, device)
        for split, (hist, seen, targets) in splits.items():
            ranks = ev.rank_cases(fn, hist, seen, targets, device)
            rows.append({"run": run, "backbone": backbone, "split": split,
                         **ev.metrics_from_ranks(ranks, KS)})
        r = rows[-1]
        print(f"{run} ({backbone}) test recall@10 {r['recall@10']:.4f} ndcg@10 {r['ndcg@10']:.4f}", flush=True)
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    summary = df.groupby(["backbone", "split"]).agg(
        runs=("run", "count"), recall10=("recall@10", "mean"), recall10_sd=("recall@10", "std"),
        ndcg10=("ndcg@10", "mean"), ndcg10_sd=("ndcg@10", "std"), recall50=("recall@50", "mean"))
    L = [f"# Proposal section 4: backbones on {name}", "",
         "Same cases, masking and metrics as reports/baselines (mean and sd over runs).", "",
         "| Backbone | Split | Runs | Recall@10 | NDCG@10 | Recall@50 |", "|---|---|---|---|---|---|"]
    for (b, s), r in summary.iterrows():
        L.append(f"| {b} | {s} | {r.runs} | {r.recall10:.4f} ({r.recall10_sd:.4f}) | "
                 f"{r.ndcg10:.4f} ({r.ndcg10_sd:.4f}) | {r.recall50:.4f} |")
    L += ["", "## Per run", "", "| Run | Backbone | Split | Recall@10 | NDCG@10 | Recall@50 |",
          "|---|---|---|---|---|---|"]
    L += [f"| {r.run} | {r.backbone} | {r.split} | {r['recall@10']:.4f} | {r['ndcg@10']:.4f} | "
          f"{r['recall@50']:.4f} |" for _, r in df.iterrows()]
    out = REPO / "reports" / "backbones"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.md").write_text("\n".join(L) + "\n")
    (out / f"{name}.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
