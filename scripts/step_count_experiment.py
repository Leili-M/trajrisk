"""Proposal section 4: does the trajectory signal depend on the number of sampling steps?

Every frozen DiffRec model (trained with 5 steps) is sampled with all 5 steps and
with evenly spaced subsets (3 steps: [4, 2, 0]; 2 steps: [4, 0]). For each setting:
  - recall@10 of the final list, and how much the final list differs from the
    5-step list (does sampling with fewer steps change the recommendation?)
  - the step-change signal: mean over non-final checkpoints of 1 - overlap@10
    between the checkpoint's list and the final list
  - its AUROC for predicting a miss, and its added value over the three controls of
    the main analysis: history length, final-score margin and mean popularity of the
    history items (5-fold cross-validation, folds split by user)
  - the overlap with the final list at each checkpoint, by normalized noise level
    (1 = noisiest step, 0 = final), so settings with different step counts can be
    compared at the same noise level

Run on the risk-validation split, so the frozen test protocol is not touched.

    sbatch --partition=gpumedium --time=01:00:00 jobs/run.sh configs/amazon_video_games.yaml \
        scripts/step_count_experiment.py --config configs/amazon_video_games.yaml \
        --runs vg_base vg_seed1 vg_seed2 vg_seed3 vg_seed4
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from trajrisk import diffrec_data as dd
from trajrisk import respaced as rs
from trajrisk import risk_eval
from trajrisk.evaluate import target_ranks
from trajrisk.trajectory import overlap_at_k, topk_lists

REPO = Path(__file__).resolve().parents[1]
K = 10


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


def load_model(cfg, run, n_items, device):
    from trajrisk.diffrec_model import build_diffrec
    run_dir = Path(cfg["paths"]["output_dir"]) / "models" / run
    with open(run_dir / "config.json") as f:
        model_cfg = json.load(f)["model"]
    model, diffusion = build_diffrec(dict(cfg, model=model_cfg), n_items, device)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model, diffusion


def run_one(model, diffusion, hist, seen, targets, subsets, device, batch_size):
    """Per-case results for every step subset."""
    out = {s: {"miss": [], "change": [], "margin": [], "same_as_full": [], "overlap": []}
           for s in subsets}
    full_steps = max(subsets)
    for idx, h in dd.iterate_batches(hist, batch_size):
        h = h.to(device)
        sn = torch.from_numpy(seen[idx].toarray()).to(device) > 0
        t = torch.as_tensor(targets[idx], device=device, dtype=torch.long)
        finals = {}
        for s, steps in subsets.items():
            traj = rs.respaced_trajectory(model, diffusion, h, steps)
            lists = topk_lists(traj, sn.unsqueeze(0).expand_as(traj), K)     # (n_ck, b, K)
            ov = torch.stack([overlap_at_k(lists[k], lists[-1]) for k in range(len(steps))])  # (n_ck, b)
            final = traj[-1].masked_fill(sn, float("-inf"))
            top11 = final.topk(K + 1, dim=-1).values
            finals[s] = lists[-1]
            d = out[s]
            d["miss"].append((target_ranks(traj[-1], t, sn) >= K).cpu().numpy())
            d["change"].append((1 - ov[:-1]).mean(0).cpu().numpy())
            d["margin"].append((-(top11[:, K - 1] - top11[:, K])).cpu().numpy())
            d["overlap"].append(ov.T.cpu().numpy())
        for s in subsets:
            out[s]["same_as_full"].append(overlap_at_k(finals[s], finals[full_steps]).cpu().numpy())
    return {s: {k: np.concatenate(v) for k, v in d.items()} for s, d in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--split", default="valid_risk")
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.config).stem

    items, inter, cases = dd.load_processed(cfg)
    part = cases[cases["split"] == args.split].reset_index(drop=True)
    hist, seen = dd.case_matrices(part, inter, len(items))
    targets = part["target_idx"].to_numpy()
    users = part["user_id"].to_numpy()
    log_hist = np.log1p(part["n_prior"].to_numpy())
    # third control of the main analysis: mean log training popularity of the history items
    tr = inter[(inter["split"] == "train") & inter["positive"]]
    log_pop = np.log1p(np.bincount(tr["item_idx"], minlength=len(items))).astype(np.float64)
    hist_pop = np.asarray(hist @ log_pop).ravel() / np.maximum(np.asarray(hist.sum(1)).ravel(), 1)

    rows, level_rows = [], []
    for run in args.runs:
        model, diffusion = load_model(cfg, run, len(items), device)
        subsets = rs.step_subsets(diffusion.steps)
        t0 = time.time()
        res = run_one(model, diffusion, hist, seen, targets, subsets, device, args.batch_size)
        for s, d in res.items():
            df = pd.DataFrame({"miss": d["miss"], "step_change@10": d["change"],
                               "small_margin": d["margin"], "log_history": log_hist,
                               "history_popularity": hist_pop})
            controls = ["log_history", "small_margin", "history_popularity"]
            base = risk_eval.cv_auroc(df, controls, users)
            both = risk_eval.cv_auroc(df, controls + ["step_change@10"], users)
            rows.append({"run": run, "steps": s, "visited": str(subsets[s]),
                         "recall@10": float(1 - d["miss"].mean()),
                         "final_list_overlap_with_5_steps": float(d["same_as_full"].mean()),
                         "mean_step_change@10": float(d["change"].mean()),
                         "auroc_step_change": float(roc_auc_score(d["miss"], d["change"])),
                         "cv_auroc_controls": base, "gain_over_controls": both - base})
            for k, tk in enumerate(subsets[s]):
                level_rows.append({"run": run, "steps": s, "checkpoint": k, "t": tk,
                                   "normalized_level": rs.normalized_level(tk, diffusion.steps),
                                   "overlap_with_final@10": float(d["overlap"][:, k].mean())})
        print(f"{run}: done in {time.time() - t0:.0f} s", flush=True)

    res = pd.DataFrame(rows)
    lev = pd.DataFrame(level_rows)
    mean = res.groupby("steps").agg(
        recall=("recall@10", "mean"), same=("final_list_overlap_with_5_steps", "mean"),
        change=("mean_step_change@10", "mean"), auroc=("auroc_step_change", "mean"),
        gain=("gain_over_controls", "mean"), gain_sd=("gain_over_controls", "std"))
    by_level = lev.pivot_table(index="normalized_level", columns="steps",
                               values="overlap_with_final@10", aggfunc="mean").sort_index(ascending=False)

    L = [f"# Proposal section 4: number of sampling steps ({name}, {args.split})", "",
         f"{len(args.runs)} models trained with 5 steps, sampled on subsets of their steps; "
         f"{len(part):,} cases. Controls: log history length, final-score margin "
         "and mean popularity of the history items, the three controls of the main analysis "
         "(5-fold CV, folds by user).", "",
         "| Steps | Recall@10 | Final list same as 5 steps | Mean step change@10 | AUROC | Gain over controls (sd) |",
         "|---|---|---|---|---|---|"]
    for s, r in mean.sort_index(ascending=False).iterrows():
        L.append(f"| {s} | {r.recall:.4f} | {100 * r.same:.1f}% | {r.change:.3f} | {r.auroc:.3f} | "
                 f"{r.gain:+.3f} ({r.gain_sd:.3f}) |")
    L += ["", "Overlap of each checkpoint's top-10 with the final top-10, by normalized noise level "
              "(1 = noisiest visited step, 0 = final; mean over models):", "",
          "| Normalized level | " + " | ".join(f"{s} steps" for s in by_level.columns) + " |",
          "|---|" + "---|" * len(by_level.columns)]
    for lvl, r in by_level.iterrows():
        L.append(f"| {lvl:.2f} | " + " | ".join("" if pd.isna(v) else f"{v:.3f}" for v in r) + " |")
    rep = REPO / "reports" / "step_counts"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"{name}.md").write_text("\n".join(L) + "\n")
    (rep / f"{name}.json").write_text(json.dumps({"per_run": rows, "by_level": level_rows}, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
