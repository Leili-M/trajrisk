"""Proposal section 4: train the second backbone (GiffCF) on our temporal split.

Same data, same catalog and same model-selection rule as DiffRec: train on the
training users, check NDCG@10 on valid_model every 10 epochs, keep the best state,
stop after `patience` epochs without improvement. Hyperparameters follow the
authors' defaults (T = 3, alpha = 1.5, embedding 200, dropout 0.5, Adam 1e-3, batch 100).

    sbatch --partition=gpumedium --time=12:00:00 jobs/run.sh configs/amazon_video_games.yaml \
        scripts/train_giffcf.py --config configs/amazon_video_games.yaml --run-name vg_giff_s1 --seed 1

Writes <output_dir>/models/<run-name>/best.pt (denoiser weights), config.json and history.json.
best.pt and config.json are rewritten at every improvement ("finished": false until the end).
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from trajrisk import diffrec_data as dd
from trajrisk import evaluate as ev
from trajrisk import giffcf as gc


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-val-cases", type=int, default=20000,
                        help="random subset of valid_model cases used for early stopping")
    for k, v in gc.HYPER.items():
        parser.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hyper = {k: getattr(args, k) for k in gc.HYPER}

    items, inter, cases = dd.load_processed(cfg)
    n_items = len(items)
    train, _ = dd.train_matrix(inter, n_items)
    part = cases[cases["split"] == "valid_model"]
    if args.max_val_cases and len(part) > args.max_val_cases:
        part = part.sample(args.max_val_cases, random_state=cfg["project"]["seed"])
    part = part.reset_index(drop=True)
    hist, seen = dd.case_matrices(part, inter, n_items)
    targets = part["target_idx"].to_numpy()

    def validate(model):
        ranks = ev.rank_cases(model.recommend, hist, seen, targets, device)
        return ev.metrics_from_ranks(ranks, [10])["ndcg@10"]

    print(f"GiffCF on {Path(args.config).stem}: {train.shape[0]:,} training users, {n_items:,} items, "
          f"{len(part):,} validation cases, device {device}", flush=True)
    t0 = time.time()
    out = Path(cfg["paths"]["output_dir"]) / "models" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    meta = {"backbone": "giffcf", "model": hyper, "seed": args.seed, "lr": args.lr,
            "batch_size": args.batch_size, "patience": args.patience, "val_every": args.val_every,
            "validation_cases": len(part)}

    def save_best(model, score, epoch):
        """Write the best model so far, so a time limit never loses the run."""
        torch.save(model.denoiser.state_dict(), out / "best.pt")
        (out / "config.json").write_text(json.dumps(dict(
            meta, best_epoch=epoch, best_val_ndcg_10=score,
            train_seconds=round(time.time() - t0), finished=False), indent=2))

    model = gc.build(train, hyper, device)
    best, epoch, history = gc.fit(model, train, validate, epochs=args.epochs,
                                  batch_size=args.batch_size, lr=args.lr, val_every=args.val_every,
                                  patience=args.patience, seed=args.seed, device=device,
                                  log=lambda s: print(s, flush=True), on_best=save_best)
    (out / "config.json").write_text(json.dumps(dict(
        meta, best_epoch=epoch, best_val_ndcg_10=best,
        train_seconds=round(time.time() - t0), finished=True), indent=2))
    (out / "history.json").write_text(json.dumps(history))
    print(f"Done: best valid NDCG@10 {best:.4f} at epoch {epoch} -> {out}")


if __name__ == "__main__":
    main()
