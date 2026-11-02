"""Proposal section 9, step 1: run the frozen fallback recommender (EASE) on the
risk-validation and test cases.

EASE uses the regularisation chosen on model-validation in section 4
(reports/baselines/<dataset>.md); it is passed explicitly and not changed afterwards.
Writes <output_dir>/fallback/ease__<split>/:
  ranks.parquet   case_id, fallback_rank (0-based rank of the target; evaluation only)
  top10.npz       the fallback's top-10 list per case (for coverage and popularity)
  manifest.json   lambda, build time, ms per case (mean and 95th percentile of batches),
                  peak GPU memory, throughput

    sbatch --partition=gpumedium --time=02:00:00 jobs/run.sh configs/amazon_books.yaml \
        scripts/fallback_ranks.py --config configs/amazon_books.yaml --lam 20
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
from trajrisk.baselines import ease, gram_matrix
from trajrisk.evaluate import target_ranks
from trajrisk.trajectory import topk_lists


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--lam", type=float, required=True, help="EASE lambda from section 4")
    parser.add_argument("--splits", nargs="+", default=["valid_risk", "test"])
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items, inter, cases = dd.load_processed(cfg)
    n_items = len(items)
    train, _ = dd.train_matrix(inter, n_items)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    score = ease(gram_matrix(train), args.lam, device)
    sync(device)
    build = time.time() - t0

    for split in args.splits:
        part = cases[cases["split"] == split].reset_index(drop=True)
        hist, seen = dd.case_matrices(part, inter, n_items)
        targets = part["target_idx"].to_numpy()
        ranks, tops, batch_ms = [], [], []
        t_all = time.time()
        with torch.no_grad():
            for idx, h in dd.iterate_batches(hist, args.batch_size):
                s = torch.from_numpy(seen[idx].toarray()).to(device) > 0
                t = torch.as_tensor(targets[idx], device=device, dtype=torch.long)
                sync(device); t1 = time.time()
                sc = score(h.to(device))
                top = topk_lists(sc, s, 10)
                sync(device)
                batch_ms.append(1000 * (time.time() - t1) / len(idx))
                ranks.append(target_ranks(sc, t, s).cpu().numpy())
                tops.append(top.cpu().numpy().astype(np.int32))
        total = time.time() - t_all
        out = Path(cfg["paths"]["output_dir"]) / "fallback" / f"ease__{split}"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"case_id": part["case_id"], "fallback_rank": np.concatenate(ranks)}) \
            .to_parquet(out / "ranks.parquet", index=False)
        np.savez(out / "top10.npz", top10=np.concatenate(tops))   # same row order as ranks.parquet
        r = np.concatenate(ranks)
        manifest = {
            "fallback": "EASE", "lambda": args.lam, "split": split, "cases": len(part),
            "build_seconds": round(build, 2),
            "ms_per_case_mean": round(float(np.average(batch_ms, weights=None)), 4),
            "ms_per_case_p95_batch": round(float(np.percentile(batch_ms, 95)), 4),
            "throughput_cases_per_s": round(len(part) / total, 1),
            "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if device == "cuda" else None,
            "recall@10": float((r < 10).mean()),
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
