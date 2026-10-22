"""Summarise uncertainty signals over several trained models (seeds), using the
success criteria of Part 13 of the report.

For every signal file (one per model):
  - cross-validated AUROC of the three controls (history length, score margin,
    history popularity) and the gain from adding the signals;
  - AUROC of each signal inside four history-length groups.
Then the mean and 95% interval over the files.

    TRAJRISK_CONFIG=configs/amazon_video_games.yaml PYTHONPATH=src \
        python scripts/seed_summary.py --split valid_model FILE [FILE ...]
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from trajrisk import diffrec_data as dd
from trajrisk import risk_eval
from trajrisk.config import load_config

CONTROLS = ["log_history", "small_margin", "history_popularity"]
NOT_SIGNALS = ({"miss", "miss_one_sample", "history_size", "user_id", "case_id",
                "date", "split", "target_idx", "n_prior"} | set(CONTROLS))


def interval(x):
    """Mean and 95% t-interval over seeds."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    if len(x) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


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
    hist_pop = np.asarray(hist @ log_pop).ravel() / n_hist
    users = part["user_id"].to_numpy()

    rows, within = [], []
    for f in args.files:
        df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
        if len(df) != len(part) or ("case_id" in df and
                                    not (df["case_id"].to_numpy() == part["case_id"].to_numpy()).all()):
            print(f"Skip {f}: rows do not match the {args.split} cases")
            continue
        df["miss"] = df["miss"].astype(bool)
        df["history_size"] = part["n_prior"].to_numpy()
        df["log_history"] = np.log1p(df["history_size"])
        df["history_popularity"] = hist_pop
        signals = [c for c in risk_eval.usable(df, list(df.columns)) if c not in NOT_SIGNALS]

        base = risk_eval.cv_auroc(df, CONTROLS, users)
        row = {"file": f, "recall@10": 1 - df["miss"].mean(), "controls": base,
               "gain_all": risk_eval.cv_auroc(df, CONTROLS + signals, users) - base}
        for s in signals:
            row[f"gain:{s}"] = risk_eval.cv_auroc(df, CONTROLS + [s], users) - base
        rows.append(row)
        w = risk_eval.within_group_table(df, signals)
        w["file"] = f
        within.append(w)

    res = pd.DataFrame(rows)
    print(f"\n{len(res)} models, split {args.split}")
    m, lo, hi = interval(res["recall@10"])
    print(f"Recall@10 of the judged list: {m:.4f}  [{lo:.4f}, {hi:.4f}]")
    m, lo, hi = interval(res["controls"])
    print(f"Controls only (CV AUROC):      {m:.3f}  [{lo:.3f}, {hi:.3f}]")

    print("\nGain over the three controls (criterion: mean >= 0.02 and interval above 0):")
    for c in [c for c in res.columns if c.startswith("gain")]:
        m, lo, hi = interval(res[c])
        ok = "PASS" if (m >= 0.02 and lo > 0) else "fail"
        print(f"  {c:28s} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {ok}")

    print("\nAUROC inside history-length groups (criterion: interval excludes 0.5 in >= 3 of 4 groups):")
    w = pd.concat(within)
    signals = [c for c in w.columns if c not in {"group", "cases", "wrong", "history_min",
                                                 "history_max", "file"}]
    for s in signals:
        parts, above, below = [], 0, 0
        for g, sub in w.groupby("group", observed=True):
            m, lo, hi = interval(sub[s])
            above += int(lo > 0.5)
            below += int(hi < 0.5)
            parts.append(f"{g} {m:.3f} [{lo:.3f},{hi:.3f}]")
        n_clear = max(above, below)
        ok = "PASS" if n_clear >= 3 else "fail"
        print(f"  {s:20s} {n_clear}/4 clear  {ok}   " + "  ".join(parts))


if __name__ == "__main__":
    main()