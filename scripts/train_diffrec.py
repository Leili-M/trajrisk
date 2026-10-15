"""Train DiffRec with regular validation and keep the best model.

    python scripts/train_diffrec.py --run-name base
    python scripts/train_diffrec.py --run-name smoke --max-epochs 2

Running the same command again resumes from the last saved checkpoint.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from trajrisk import diffrec_data as dd
from trajrisk.config import load_config
from trajrisk.diffrec_model import build_diffrec, recommend_scores
from trajrisk.evaluate import metrics_from_ranks, rank_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="base")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs
    tr, seed = cfg["train"], cfg["project"]["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    run_dir = Path(cfg["paths"]["output_dir"]) / "models" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # Data
    items, inter, cases = dd.load_processed(cfg)
    n_items = len(items)
    X_train, _ = dd.train_matrix(inter, n_items)
    vm = cases[cases["split"] == "valid_model"].reset_index(drop=True)
    H_vm, S_vm = dd.case_matrices(vm, inter, n_items)
    t_vm = vm["target_idx"].to_numpy()

    # Model
    model, diffusion = build_diffrec(cfg, n_items, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["lr"],
                                 weight_decay=tr["weight_decay"])
    state = {"epoch": 0, "history": [], "best_score": -1.0, "best_epoch": 0, "bad_evals": 0}

    # Resume
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        diffusion.Lt_history = ck["Lt_history"].to(device)
        diffusion.Lt_count = ck["Lt_count"].to(device)
        torch.set_rng_state(ck["rng_cpu"])
        if device == "cuda" and ck["rng_cuda"] is not None:
            torch.cuda.set_rng_state(ck["rng_cuda"])
        state = ck["state"]
        print(f"Resumed after epoch {state['epoch']}")

    metric = tr["select_metric"]
    print(f"Device: {device}   recipes: {n_items:,}   training users: {X_train.shape[0]:,}   "
          f"valid_model cases: {len(vm):,}")

    while state["epoch"] < tr["max_epochs"] and state["bad_evals"] < tr["patience"]:
        epoch = state["epoch"] + 1
        model.train()
        t0, losses = time.time(), []
        for _, batch in dd.iterate_batches(X_train, tr["batch_size"], shuffle=True, seed=seed + epoch):
            loss = diffusion.training_losses(model, batch.to(device),
                                             reweight=cfg["model"]["reweight"])["loss"].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        state["epoch"] = epoch
        row = {"epoch": epoch, "loss": round(float(np.mean(losses)), 5),
               "seconds": round(time.time() - t0, 1)}

        is_eval = epoch % tr["eval_every"] == 0 or epoch == tr["max_epochs"]
        if is_eval:
            model.eval()
            ranks = rank_cases(lambda h: recommend_scores(model, diffusion, h),
                               H_vm, S_vm, t_vm, device)
            row.update({k: round(v, 5) for k, v in
                        metrics_from_ranks(ranks, cfg["evaluation"]["ks"]).items()})
            if row[metric] > state["best_score"]:
                state.update(best_score=row[metric], best_epoch=epoch, bad_evals=0)
                torch.save(model.state_dict(), run_dir / "best.pt")
                row["best"] = True
            else:
                state["bad_evals"] += 1

        state["history"].append(row)
        print(json.dumps(row), flush=True)

        if is_eval:   # checkpoint after every evaluation
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "Lt_history": diffusion.Lt_history.cpu(), "Lt_count": diffusion.Lt_count.cpu(),
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state() if device == "cuda" else None,
                "state": state,
            }, last)
            with open(run_dir / "history.json", "w") as f:
                json.dump(state["history"], f, indent=2)

    reason = "early stop" if state["bad_evals"] >= tr["patience"] else "max epochs"
    summary = {"stopped_by": reason, "epochs": state["epoch"],
               "best_epoch": state["best_epoch"], f"best_{metric}": state["best_score"]}
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Done:", json.dumps(summary))


if __name__ == "__main__":
    main()
