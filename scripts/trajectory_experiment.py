"""Stage 5: noisy, random trajectories as uncertainty signals.

Loads a trained DiffRec run, computes the signals for every case of one split,
and runs the Part 10 checks (history length, within-group AUROC, added value).

    TRAJRISK_CONFIG=configs/movielens.yaml PYTHONPATH=src \
        python scripts/trajectory_experiment.py --run-name ml_noise_mid --samples 10

    # Part 10 setting (one noise-free trajectory) for the same model:
    ... --samples 1 --no-noisy-start --no-sampling-noise

Results go to <output_dir>/stochastic/<run-name>__<setting>/.
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
from trajrisk.stochastic import case_signals, noise_level

STEP_SIGNALS = ["step_change@10", "step_change@50", "step_score_change"]
SAMPLE_SIGNALS = ["sample_disagree@10", "sample_disagree@50", "sample_rank_spread"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="valid_model",
                        help="valid_model = same cases as Part 10; valid_risk = held-out check")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--no-noisy-start", action="store_true")
    parser.add_argument("--no-sampling-noise", action="store_true")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--max-cases", type=int, default=None, help="for a quick smoke test")
    args = parser.parse_args()

    noisy_start, sampling_noise = not args.no_noisy_start, not args.no_sampling_noise
    setting = (f"S{args.samples}_" + ("noisystart" if noisy_start else "cleanstart")
               + ("_sampling" if sampling_noise else "_nosampling"))

    cfg = load_config()
    seed = cfg["project"]["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The model settings come from the run itself, so the noise matches its training
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

    level = noise_level(diffusion)
    print(f"Run {args.run_name}: steps={diffusion.steps}  noise_scale={cfg['model']['noise_scale']}  "
          f"noise_max={cfg['model']['noise_max']}")
    print(f"  noise std at the last step: {level['noise_std_last_step']:.4f} "
          f"(history values are 0 or 1)")
    print(f"Setting {setting}: {len(part):,} {args.split} cases, device {device}")

    torch.manual_seed(seed)
    collected, t0 = [], time.time()
    for idx, h in dd.iterate_batches(hist, args.batch_size):
        s = torch.from_numpy(seen[idx].toarray()).to(device) > 0
        t = torch.as_tensor(targets[idx], device=device, dtype=torch.long)
        collected.append(case_signals(model, diffusion, h.to(device), s, t, args.samples,
                                      noisy_start, sampling_noise))
    sig = {k: torch.cat([c[k] for c in collected]).numpy() for k in collected[0]}
    print(f"  signals computed in {time.time() - t0:.0f} s")

    df = pd.DataFrame(sig)
    df["miss"] = df["miss"] > 0.5
    df["miss_one_sample"] = df["miss_one_sample"] > 0.5
    df["history_size"] = part["n_prior"].to_numpy()
    df["user_id"] = part["user_id"].to_numpy()
    df["case_id"] = part["case_id"].to_numpy()

    sets = {"steps": STEP_SIGNALS, "samples": SAMPLE_SIGNALS, "margin": ["small_margin"],
            "all": STEP_SIGNALS + SAMPLE_SIGNALS + ["small_margin"]}
    res = risk_eval.evaluate(df, sets, df["user_id"].to_numpy())

    summary = {
        "run": args.run_name, "setting": setting, "split": args.split, "cases": len(df),
        "steps": diffusion.steps, "noise_scale": cfg["model"]["noise_scale"],
        "noise_max": cfg["model"]["noise_max"], **level,
        "samples": args.samples, "noisy_start": noisy_start, "sampling_noise": sampling_noise,
        "recall@10_averaged": float(1 - df["miss"].mean()),
        "recall@10_one_sample": float(1 - df["miss_one_sample"].mean()),
        "mean_sample_disagree@10": float(df["sample_disagree@10"].mean()),
        "mean_step_change@10": float(df["step_change@10"].mean()),
    }
    av = res["added_value"].set_index("predictors")["cv_auroc"]
    summary["cv_auroc_history_only"] = float(av["history only"])
    summary["cv_auroc_history_plus_all"] = float(av.get("history + all", np.nan))
    wg = res["within_groups"]
    sig_cols = [c for c in wg.columns if c not in ("group", "cases", "wrong",
                                                    "history_min", "history_max")]
    summary["mean_within_group_auroc"] = {c: float(wg[c].mean()) for c in sig_cols}

    out = Path(cfg["paths"]["output_dir"]) / "stochastic" / f"{args.run_name}__{setting}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "signals.csv", index=False)
    for name, table in res.items():
        table.to_csv(out / f"{name}.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.set_option("display.width", 200)
    print(f"\nRecall@10 (averaged scores): {summary['recall@10_averaged']:.4f}   "
          f"one sample: {summary['recall@10_one_sample']:.4f}")
    print(f"Mean disagreement between samples (top-10): {summary['mean_sample_disagree@10']:.3f}")
    print(f"Mean change along the steps (top-10):       {summary['mean_step_change@10']:.3f}")
    print("\n1) Each signal alone (0.5 = chance) and its correlation with history length:")
    print(res["single"].round(3).to_string(index=False))
    print("\n2) AUROC inside history-length groups (about 0.5 everywhere = no extra information):")
    print(wg.round(3).to_string(index=False))
    print("\n3) Does a signal add anything to history length? (5-fold CV, split by user)")
    print(res["added_value"].round(3).to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
