"""Added value of uncertainty signals over ALL simple controls (Part 13 of the report):
history length, score margin, and the popularity of the user's history items.

Reads signal files saved by trajectory_experiment.py / masking_experiment.py.

    TRAJRISK_CONFIG=configs/amazon_video_games.yaml PYTHONPATH=src python scripts/control_check.py FILE [FILE ...]
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from trajrisk import diffrec_data as dd
from trajrisk import risk_eval
from trajrisk.config import load_config

CONTROLS = ["log_history", "small_margin", "history_popularity"]
NOT_SIGNALS = ({"miss", "miss_one_sample", "history_size", "user_id", "case_id",
                "date", "split", "target_idx", "n_prior"} | set(CONTROLS))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--split", default="valid_model")
    args = parser.parse_args()

    cfg = load_config()
    items, inter, cases = dd.load_processed(cfg)
    n_items = len(items)
    train, _ = dd.train_matrix(inter, n_items)
    log_pop = np.log1p(np.asarray(train.sum(axis=0)).ravel())

    part = cases[cases["split"] == args.split].reset_index(drop=True)
    hist, _ = dd.case_matrices(part, inter, n_items)
    n_hist = np.maximum(np.asarray(hist.sum(axis=1)).ravel(), 1)
    # Average (log) training popularity of the items in the user's history
    hist_pop = np.asarray(hist @ log_pop).ravel() / n_hist
    users = part["user_id"].to_numpy()

    for f in args.files:
        df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
        if len(df) != len(part):
            print(f"\nSkip {f}: {len(df)} rows, expected {len(part)}")
            continue
        if "case_id" in df and not (df["case_id"].to_numpy() == part["case_id"].to_numpy()).all():
            print(f"\nSkip {f}: cases are in a different order")
            continue
        df["miss"] = df["miss"].astype(bool)
        df["log_history"] = np.log1p(part["n_prior"].to_numpy())
        df["history_popularity"] = hist_pop
        signals = [c for c in risk_eval.usable(df, list(df.columns)) if c not in NOT_SIGNALS]

        print(f"\n=== {f}")
        print(f"history popularity alone: AUROC {roc_auc_score(df['miss'], hist_pop):.3f}")
        base = risk_eval.cv_auroc(df, CONTROLS, users)
        print(f"controls only (history length + margin + history popularity): {base:.3f}")
        for s in signals:
            a = risk_eval.cv_auroc(df, CONTROLS + [s], users)
            rho = spearmanr(df[s], hist_pop).correlation
            print(f"  + {s:20s} {a:.3f}  gain {a - base:+.3f}   corr with history popularity {rho:+.2f}")
        a = risk_eval.cv_auroc(df, CONTROLS + signals, users)
        print(f"  + all signals          {a:.3f}  gain {a - base:+.3f}")


if __name__ == "__main__":
    main()