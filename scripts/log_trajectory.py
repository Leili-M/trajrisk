"""Proposal section 4: log the denoising trajectory of a frozen DiffRec model.

For one model and one or more splits this writes, under
<output_dir>/trajlogs/<run>__<split>/ :
  trajectory.npz   top-M items and scores per case and checkpoint, the distance of
                   every checkpoint to the final scores (full vectors), final norm
  cases.parquet    case_id, user_id, query time (row order of trajectory.npz)
  checkpoints.json checkpoint index, diffusion step t, noise level, normalized step
  labels.parquet   SEPARATE evaluation table: target item and its rank at every
                   checkpoint (never read by the signal code)
  manifest.json    run, checkpoint file hash, configuration hash, catalog and split
                   hashes, code commit, seed, top-M, runtime breakdown

    sbatch --partition=gpumedium --time=01:00:00 jobs/run.sh configs/amazon_video_games.yaml \
        scripts/log_trajectory.py --config configs/amazon_video_games.yaml \
        --runs vg_base vg_seed1 vg_seed2 vg_seed3 vg_seed4 --splits valid_model valid_risk test
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajrisk import diffrec_data as dd
from trajrisk import trajlog as tl
from trajrisk.trajectory import diffrec_trajectory, overlap_at_k, topk_lists

REPO = Path(__file__).resolve().parents[1]


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def load_model(cfg, run, n_items, device, train=None):
    """Backbone of a saved run: trajectory function, checkpoint table, settings, weights file.

    DiffRec runs log the predicted clean vector at every reverse step and the noise
    level of each step. GiffCF runs (config.json "backbone": "giffcf") log the
    predicted clean vector at t = T, ..., 1 and the final output, with the smoothing
    level of each step.
    """
    run_dir = Path(cfg["paths"]["output_dir"]) / "models" / run
    with open(run_dir / "config.json") as f:
        meta = json.load(f)
    if meta.get("backbone") == "giffcf":
        from trajrisk import giffcf as gc
        model, _ = gc.load_run(run_dir, train, device)
        return {"name": "GiffCF", "trajectory": model.trajectory, "checkpoints": model.checkpoints(),
                "model_cfg": meta["model"], "ckpt": run_dir / "best.pt"}
    from trajrisk.diffrec_model import build_diffrec
    model_cfg = meta["model"]
    model, diffusion = build_diffrec(dict(cfg, model=model_cfg), n_items, device)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return {"name": "DiffRec", "trajectory": lambda h: diffrec_trajectory(model, diffusion, h),
            "checkpoints": tl.noise_schedule(diffusion), "model_cfg": model_cfg,
            "ckpt": run_dir / "best.pt"}


def check_top_m(traj, seen, logged_items, k=10):
    """Lists of length k <= M from the log must equal lists from the full scores."""
    full = topk_lists(traj, seen.unsqueeze(0).expand_as(traj), k)          # (n_ck, b, k)
    from_log = torch.as_tensor(logged_items[:, :, :k]).permute(1, 0, 2).to(full.device)
    return bool((overlap_at_k(full, from_log.long()) == 1).all())


def log_split(cfg, run, split, backbone, data, args, device):
    items, inter, cases = data
    n_items = len(items)
    part = cases[cases["split"] == split].reset_index(drop=True)
    hist, seen = dd.case_matrices(part, inter, n_items)
    targets = part["target_idx"].to_numpy()
    torch.manual_seed(cfg["project"]["seed"])

    t_model = t_sort = t_store = 0.0
    chunks, rank_chunks, checked = [], [], None
    t_all = time.time()
    for idx, h in dd.iterate_batches(hist, args.batch_size):
        s = torch.from_numpy(seen[idx].toarray()).to(device) > 0
        t = torch.as_tensor(targets[idx], device=device, dtype=torch.long)
        sync(device); t0 = time.time()
        traj = backbone["trajectory"](h.to(device))
        sync(device); t1 = time.time()
        logged, ranks = tl.log_batch(traj, s, t, args.top_m)
        sync(device); t2 = time.time()
        if checked is None:
            checked = check_top_m(traj, s, logged["items"])
        chunks.append(logged)
        rank_chunks.append(ranks)
        t_model += t1 - t0
        t_sort += t2 - t1

    out = Path(cfg["paths"]["output_dir"]) / "trajlogs" / f"{run}__{split}"
    out.mkdir(parents=True, exist_ok=True)
    t3 = time.time()
    arrays = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    np.savez(out / "trajectory.npz", **arrays)
    part[["case_id", "user_id", "date"]].rename(columns={"date": "query_time"}) \
        .to_parquet(out / "cases.parquet", index=False)
    ranks = np.concatenate(rank_chunks)
    labels = pd.DataFrame({"case_id": part["case_id"], "target_idx": targets})
    for k in range(ranks.shape[1]):
        labels[f"target_rank_ck{k}"] = ranks[:, k]
    labels["target_rank_final"] = ranks[:, -1]
    labels.to_parquet(out / "labels.parquet", index=False)
    schedule = backbone["checkpoints"]
    (out / "checkpoints.json").write_text(json.dumps(schedule, indent=2))
    t_store = time.time() - t3
    total = time.time() - t_all

    data_dir = Path(cfg["paths"]["data_dir"]) / "processed"
    manifest = {
        "run": run, "split": split, "cases": int(len(part)), "top_m": args.top_m,
        "checkpoints": len(schedule), "seed": cfg["project"]["seed"],
        "sampling": "no added noise at the start, no sampling noise (deterministic)",
        "backbone": backbone["name"],
        "model_checkpoint": str(backbone["ckpt"]), "model_checkpoint_sha256": sha256_file(backbone["ckpt"]),
        "model_config_sha256": sha256_json(backbone["model_cfg"]), "model_config": backbone["model_cfg"],
        "data_config_sha256": sha256_json(cfg["data"]),
        "catalog_sha256": sha256_file(data_dir / "items.parquet"),
        "cases_sha256": sha256_file(data_dir / "cases.parquet"),
        "split_boundaries": cfg["data"]["split"], "code_commit": git_commit(),
        "lists_up_to_k10_match_full_scores": checked,
        "recall@10_final": float((ranks[:, -1] < 10).mean()),
        "runtime_seconds": {"trajectory": round(t_model, 2), "sorting_and_summaries": round(t_sort, 2),
                            "storage": round(t_store, 2), "total": round(total, 2),
                            "ms_per_case": round(1000 * total / max(len(part), 1), 3)},
        "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "torch": torch.__version__,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["valid_model", "valid_risk", "test"])
    parser.add_argument("--top-m", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = dd.load_processed(cfg)
    train, _ = dd.train_matrix(data[1], len(data[0]))
    rows = []
    for run in args.runs:
        backbone = load_model(cfg, run, len(data[0]), device, train)
        for split in args.splits:
            m = log_split(cfg, run, split, backbone, data, args, device)
            rows.append({"run": run, "backbone": backbone["name"], "split": split, "cases": m["cases"],
                         "recall@10": m["recall@10_final"],
                         "lists_match_full": m["lists_up_to_k10_match_full_scores"],
                         **{f"sec_{k}": v for k, v in m["runtime_seconds"].items()}})
            print(f"{run} {split}: {m['cases']:,} cases, recall@10 {m['recall@10_final']:.4f}, "
                  f"lists match {m['lists_up_to_k10_match_full_scores']}, "
                  f"{m['runtime_seconds']['ms_per_case']} ms/case", flush=True)
    name = Path(args.config).stem
    rep = REPO / "reports" / "trajlogs"
    rep.mkdir(parents=True, exist_ok=True)
    # one summary file per dataset; GiffCF runs go to a separate file so the DiffRec
    # summary of step B is not overwritten
    suffix = "" if all(r["backbone"] == "DiffRec" for r in rows) else "_giffcf"
    pd.DataFrame(rows).to_csv(rep / f"{name}{suffix}.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
