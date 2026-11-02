"""Proposal section 7: repeated-sampling baseline (2, 5, 10 samples) and masked histories.

For every DiffRec run and split this writes
<output_dir>/trajlogs/<run>__<split>/perturb.parquet (case_id + signals, no labels) and
perturb_manifest.json. The judged list is the ordinary DiffRec list; the script checks
that it equals the final list stored in the trajectory log, so the labels of the log
apply unchanged. scripts/judge.py picks the file up automatically.

    sbatch --partition=gpumedium --time=02:00:00 jobs/run.sh configs/amazon_video_games.yaml \
        scripts/perturbation_signals.py --config configs/amazon_video_games.yaml \
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

from trajrisk import diffrec_data as dd
from trajrisk.perturb import perturbation_signals


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


def load_diffrec(cfg, run, n_items, device):
    run_dir = Path(cfg["paths"]["output_dir"]) / "models" / run
    meta = json.loads((run_dir / "config.json").read_text())
    if meta.get("backbone") == "giffcf":
        raise SystemExit(f"{run}: GiffCF has no sampling noise; this script is for DiffRec runs.")
    from trajrisk.diffrec_model import build_diffrec
    model, diffusion = build_diffrec(dict(cfg, model=meta["model"]), n_items, device)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model, diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["valid_risk", "test"])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--masks", type=int, default=10)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items, inter, cases = dd.load_processed(cfg)
    n_items = len(items)

    for run in args.runs:
        model, diffusion = load_diffrec(cfg, run, n_items, device)
        for split in args.splits:
            out = Path(cfg["paths"]["output_dir"]) / "trajlogs" / f"{run}__{split}"
            part = cases[cases["split"] == split].reset_index(drop=True)
            hist, seen = dd.case_matrices(part, inter, n_items)
            torch.manual_seed(cfg["project"]["seed"])
            chunks, tops, t0 = [], [], time.time()
            for idx, h in dd.iterate_batches(hist, args.batch_size):
                s = torch.from_numpy(seen[idx].toarray()).to(device) > 0
                sig, top = perturbation_signals(model, diffusion, h.to(device), s,
                                                args.samples, args.masks, args.rate)
                chunks.append(sig)
                tops.append(top.numpy())
            if device == "cuda":
                torch.cuda.synchronize()
            seconds = time.time() - t0
            df = pd.DataFrame({k: torch.cat([c[k] for c in chunks]).numpy() for k in chunks[0]})
            df.insert(0, "case_id", part["case_id"].to_numpy())

            # the judged list must be the final list of the trajectory log
            logged = pd.read_parquet(out / "cases.parquet")["case_id"].to_numpy()
            with np.load(out / "trajectory.npz") as z:
                final = pd.DataFrame(z["items"][:, -1, :10], index=logged)
            mine = np.concatenate(tops)
            same = (np.sort(final.loc[df["case_id"]].to_numpy(), 1) == np.sort(mine, 1)).all(1)
            if same.mean() < 0.999:
                raise RuntimeError(f"{run} {split}: only {same.mean():.4f} of lists match the log")

            df.to_parquet(out / "perturb.parquet", index=False)
            manifest = {"run": run, "split": split, "cases": len(df), "samples": args.samples,
                        "masks": args.masks, "mask_rate": args.rate,
                        "seed": cfg["project"]["seed"], "lists_match_log": float(same.mean()),
                        "noise_scale": diffusion.noise_scale, "seconds": round(seconds, 1),
                        "ms_per_case": round(1000 * seconds / max(len(df), 1), 3)}
            (out / "perturb_manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"{run} {split}: {len(df):,} cases, lists match {same.mean():.4f}, "
                  f"{manifest['ms_per_case']} ms/case, mean samp10_set "
                  f"{df['samp10_set'].mean():.3f}, mean mask_set {df['mask_set'].mean():.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
