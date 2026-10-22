"""RQ3 'why': shorten MovieLens histories and test whether the signals become informative.

The trained model is unchanged; only its input is cut to the last m liked items before
each case. Signals, controls and the judged list all use the shortened input.

    sbatch jobs/run.sh configs/movielens.yaml scripts/truncation_experiment.py --run-name ml_seed1

Results: <output_dir>/truncation/<run-name>_<split>.csv (one row per m)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajrisk import diffrec_data as dd
from trajrisk import risk_eval
from trajrisk.config import load_config
from trajrisk.diffrec_model import build_diffrec
from trajrisk.masking import masked_signals
from trajrisk.trajectory import diffrec_trajectory, overlap_at_k, topk_lists
from trajrisk.truncate import last_m_history, random_m_history

CONTROLS = ["log_history", "small_margin", "history_popularity"]
TRAJ = ["step_change@10", "step_change@50", "step_score_change"]
MASK = ["mask_change@10", "mask_change@50", "mask_score_change", "mask_unstable@10"]


@torch.no_grad()
def trajectory_signals(model, diffusion, h, seen):
    traj = diffrec_trajectory(model, diffusion, h)
    t10 = torch.stack([topk_lists(g, seen, 10) for g in traj])
    t50 = torch.stack([topk_lists(g, seen, 50) for g in traj])
    change = (traj[0] - traj[-1]).norm(dim=-1) / (traj[-1].norm(dim=-1) + 1e-8)
    return {"step_change@10": (1 - overlap_at_k(t10[:-1], t10[-1]).mean(0)).cpu(),
            "step_change@50": (1 - overlap_at_k(t50[:-1], t50[-1]).mean(0)).cpu(),
            "step_score_change": change.cpu()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="valid_model")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 0],
                        help="keep the last m liked items; 0 = full history")
    parser.add_argument("--order", choices=["recent", "random"], default="recent")
    parser.add_argument("--masks", type=int, default=10)
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["project"]["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(cfg["paths"]["output_dir"]) / "models" / args.run_name
    with open(run_dir / "config.json") as f:
        cfg["model"] = json.load(f)["model"]

    items, inter, cases = dd.load_processed(cfg)
    n_items = len(items)
    model, diffusion = build_diffrec(cfg, n_items, device)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    part = cases[cases["split"] == args.split].reset_index(drop=True)
    full_hist, seen = dd.case_matrices(part, inter, n_items)
    train, _ = dd.train_matrix(inter, n_items)
    log_pop = np.log1p(np.asarray(train.sum(axis=0)).ravel())
    targets = part["target_idx"].to_numpy()
    users = part["user_id"].to_numpy()

    rows = []
    for m in args.lengths:
        t0 = time.time()
        if m == 0:
            hist = full_hist
        elif args.order == "recent":
            hist = last_m_history(part, inter, n_items, m)
        else:
            hist = random_m_history(part, inter, n_items, m, seed)
        torch.manual_seed(seed)
        collected = []
        for idx, h in dd.iterate_batches(hist, args.batch_size):
            s = torch.from_numpy(seen[idx].toarray()).to(device) > 0
            t = torch.as_tensor(targets[idx], device=device, dtype=torch.long)
            h = h.to(device)
            sig = masked_signals(model, diffusion, h, s, t, args.masks, args.rate)
            sig.update(trajectory_signals(model, diffusion, h, s))
            collected.append(sig)
        df = pd.DataFrame({k: torch.cat([c[k] for c in collected]).numpy() for k in collected[0]})
        df["miss"] = df["miss"] > 0.5
        n_hist = np.asarray(hist.sum(axis=1)).ravel()
        df["log_history"] = np.log1p(n_hist)
        df["history_popularity"] = np.asarray(hist @ log_pop).ravel() / np.maximum(n_hist, 1)

        base = risk_eval.cv_auroc(df, CONTROLS, users)
        mask_cols = risk_eval.usable(df, MASK)
        row = {"run": args.run_name, "split": args.split, "order": args.order, "m": m if m else "full",
               "median_history": float(np.median(n_hist)),
               "recall@10": float(1 - df["miss"].mean()), "controls": base,
               "gain_trajectory": risk_eval.cv_auroc(df, CONTROLS + TRAJ, users) - base,
               "gain_masking": (risk_eval.cv_auroc(df, CONTROLS + mask_cols, users) - base
                                if mask_cols else float("nan"))}
        rows.append(row)
        print(f"m={row['m']!s:>4}  median history {row['median_history']:5.0f}  "
              f"recall@10 {row['recall@10']:.4f}  controls {base:.3f}  "
              f"gain trajectory {row['gain_trajectory']:+.3f}  "
              f"gain masking {row['gain_masking']:+.3f}  ({time.time() - t0:.0f} s)")

    out = Path(cfg["paths"]["output_dir"]) / "truncation"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{args.run_name}_{args.split}{'' if args.order == 'recent' else '_random'}.csv", index=False)
    print("Saved to", out / f"{args.run_name}_{args.split}{'' if args.order == 'recent' else '_random'}.csv")


if __name__ == "__main__":
    main()