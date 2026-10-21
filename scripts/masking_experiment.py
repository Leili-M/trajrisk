"""Phase 3: uncertainty from masked histories (see src/trajrisk/masking.py).

    sbatch jobs/run.sh configs/movielens.yaml scripts/masking_experiment.py --run-name ml_base

Results go to <output_dir>/masking/<run-name>__r<rate>/
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from trajrisk import diffrec_data as dd
from trajrisk import risk_eval
from trajrisk.config import load_config
from trajrisk.diffrec_model import build_diffrec
from trajrisk.masking import masked_signals

MASK_SIGNALS = ["mask_change@10", "mask_change@50", "mask_score_change", "mask_unstable@10"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="valid_model")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--masks", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--max-cases", type=int, default=None, help="for a quick test")
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["project"]["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use the model settings the run was trained with
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
    if args.max_cases:
        part = part.sample(min(args.max_cases, len(part)), random_state=seed).reset_index(drop=True)
    hist, seen = dd.case_matrices(part, inter, n_items)
    targets = part["target_idx"].to_numpy()
    users = part["user_id"].to_numpy()
    print(f"Run {args.run_name}: {len(part):,} {args.split} cases, "
          f"{args.masks} masks per case, device {device}")

    for rate in args.rates:
        torch.manual_seed(seed)
        t0 = time.time()
        collected = []
        for idx, h in dd.iterate_batches(hist, args.batch_size):
            s = torch.from_numpy(seen[idx].toarray()).to(device) > 0
            t = torch.as_tensor(targets[idx], device=device, dtype=torch.long)
            collected.append(masked_signals(model, diffusion, h.to(device), s, t,
                                            args.masks, rate))
        df = pd.DataFrame({k: torch.cat([c[k] for c in collected]).numpy()
                           for k in collected[0]})
        df["miss"] = df["miss"] > 0.5
        df["history_size"] = part["n_prior"].to_numpy()

        res = risk_eval.evaluate(df, {"masking": MASK_SIGNALS, "margin": ["small_margin"]}, users)
        added = res["added_value"].set_index("predictors")

        print("\n" + "=" * 90)
        print(f"Mask rate {rate:.0%}   ({time.time() - t0:.0f} s)")
        print(f"Recall@10 (full history): {1 - df['miss'].mean():.4f}")
        print(f"Mean change of the top-10 when masking: {df['mask_change@10'].mean():.3f}")
        print("\n1) Each signal alone (0.5 = chance) and its correlation with history length:")
        print(res["single"].round(3).to_string(index=False))
        print("\n2) AUROC inside history-length groups:")
        print(res["within_groups"].round(3).to_string(index=False))
        print("\n3) Added value over history length (5-fold CV, split by user):")
        print(res["added_value"].round(3).to_string(index=False))

        out = Path(cfg["paths"]["output_dir"]) / "masking" / f"{args.run_name}__r{int(rate * 100)}"
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "signals.parquet")
        for name, table in res.items():
            table.to_csv(out / f"{name}.csv", index=False)
        summary = {
            "run": args.run_name, "split": args.split, "rate": rate, "masks": args.masks,
            "cases": len(df), "recall@10": float(1 - df["miss"].mean()),
            "mean_mask_change@10": float(df["mask_change@10"].mean()),
            "cv_auroc_history_only": float(added.loc["history only", "cv_auroc"]),
            "cv_auroc_history_plus_masking": (float(added.loc["history + masking", "cv_auroc"])
                                              if "history + masking" in added.index else None),
        }
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()