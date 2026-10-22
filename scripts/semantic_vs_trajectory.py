"""Does the trajectory still add value once the simple TEXT signals are known?

The text signals list_spread (how varied the top-10 is in meaning) and
hist_to_list (how far the top-10 is from the history in meaning) need no
trajectory. Here they are treated as extra controls, and we ask whether the
trajectory and masking signals still add AUROC on top of them.

    sbatch jobs/run.sh configs/amazon_video_games.yaml scripts/semantic_vs_trajectory.py --runs vg_base vg_seed1 vg_seed2 vg_seed3 vg_seed4
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from trajrisk import risk_eval
from trajrisk.config import load_config

CONTROLS = ["log_history", "small_margin", "history_popularity"]
TEXT = ["list_spread", "hist_to_list"]
TRAJ = ["step_change@10", "step_change@50", "step_score_change"]
MASK = ["mask_change@10", "mask_change@50", "mask_score_change", "mask_unstable@10"]
SETS = {
    "controls": CONTROLS,
    "controls + text": CONTROLS + TEXT,
    "controls + trajectory": CONTROLS + TRAJ,
    "controls + masking": CONTROLS + MASK,
    "controls + text + trajectory": CONTROLS + TEXT + TRAJ,
    "controls + text + masking": CONTROLS + TEXT + MASK,
    "controls + text + both": CONTROLS + TEXT + TRAJ + MASK,
}
GAINS = [("text over controls", "controls + text", "controls"),
         ("trajectory over controls+text", "controls + text + trajectory", "controls + text"),
         ("masking over controls+text", "controls + text + masking", "controls + text"),
         ("both over controls+text", "controls + text + both", "controls + text"),
         ("text over controls+both", "controls + text + both", None)]


def load_run(out, run, split, rate):
    tag = "" if split == "valid_model" else f"_{split}"
    sem = pd.read_parquet(out / "semantic" / f"{run}{tag}" / "signals.parquet")
    traj = pd.read_csv(out / "stochastic" / f"{run}__S1_cleanstart_nosampling{tag}" / "signals.csv")
    mask = pd.read_parquet(out / "masking" / f"{run}__r{rate}{tag}" / "signals.parquet")
    assert len(sem) == len(traj) == len(mask), (len(sem), len(traj), len(mask))
    # the three files must list the cases in the same order: the misses must agree
    for name, other in [("trajectory", traj), ("masking", mask)]:
        agree = (sem["miss"].astype(bool).to_numpy() == other["miss"].astype(bool).to_numpy()).mean()
        assert agree > 0.999, f"{name} file is not aligned ({agree:.3%} of misses agree)"
    df = sem.copy()
    for c in TRAJ:
        df[c] = traj[c].to_numpy()
    for c in MASK:
        df[c] = mask[c].to_numpy()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--rate", type=int, default=50)
    parser.add_argument("--splits", nargs="+", default=["valid_model", "valid_risk"])
    args = parser.parse_args()
    out = Path(load_config()["paths"]["output_dir"])

    for split in args.splits:
        rows = []
        for run in args.runs:
            df = load_run(out, run, split, args.rate)
            users = df["user_id"].to_numpy()
            auc = {k: roc_auc_score(df["miss"], risk_eval.cv_predict(df, cols, users))
                   for k, cols in SETS.items()}
            auc["controls + both"] = roc_auc_score(
                df["miss"], risk_eval.cv_predict(df, CONTROLS + TRAJ + MASK, users))
            row = {"run": run}
            for name, a, b in GAINS:
                row[name] = auc[a] - auc[b if b else "controls + both"]
            rows.append(row)
        res = pd.DataFrame(rows)
        print("\n" + "=" * 80 + f"\n{split}: AUROC gain, one row per seed")
        print(res.round(3).to_string(index=False))
        print("\nMean and 95% interval over seeds:")
        for name, _, _ in GAINS:
            v = res[name].to_numpy()
            half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            print(f"   {name:32s} {v.mean():+.3f}  [{v.mean() - half:+.3f}, {v.mean() + half:+.3f}]")
        res.to_csv(out / "semantic" / f"vs_trajectory_{split}.csv", index=False)


if __name__ == "__main__":
    main()
