"""Proposal section 5: uncertainty signals from the trajectory logs.

For every run and split this reads <output_dir>/trajlogs/<run>__<split>/ and writes
signals.parquet next to it: case_id, user_id, every section-5 signal and the controls
(log_history, history_popularity, rec_popularity). The label file is NOT read here.

Afterwards, on ONE split only (default valid_risk), the labels are joined in memory to
report the AUROC of each signal on its own and to choose the weighting (rho) and the
entropy temperature (tau). The test split is never used for any choice.

    sbatch --partition=gpumedium --time=01:00:00 jobs/run.sh configs/amazon_video_games.yaml \
        scripts/compute_signals.py --config configs/amazon_video_games.yaml \
        --runs vg_base vg_seed1 vg_seed2 vg_seed3 vg_seed4
Report: reports/signals/<dataset>[_giffcf].md and .json
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from trajrisk import diffrec_data as dd
from trajrisk import risk_eval
from trajrisk import signals as sg

REPO = Path(__file__).resolve().parents[1]
K = 10


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


def item_counts(train):
    """Positive training interactions per item (scipy sparse or torch tensor)."""
    try:
        import torch
        if torch.is_tensor(train):
            t = train.coalesce() if train.is_sparse else train
            s = torch.sparse.sum(t, dim=0).to_dense() if t.is_sparse else t.sum(dim=0)
            return s.cpu().numpy().astype(float).ravel()
    except ImportError:
        pass
    return np.asarray(train.sum(axis=0), dtype=float).ravel()


def log_dir(cfg, run, split):
    return Path(cfg["paths"]["output_dir"]) / "trajlogs" / f"{run}__{split}"


def score_scale(cfg, run):
    """Std of the final top-M scores on model-validation cases (no labels involved)."""
    with np.load(log_dir(cfg, run, "valid_model") / "trajectory.npz") as z:
        return float(np.std(z["scores"][:, -1, :]))


def controls(part, inter, n_items, log_pop, final_items):
    """Three controls of the risk model. part: cases of one split from load_processed."""
    hist, _ = dd.case_matrices(part, inter, n_items)
    n_prior = part["n_prior"].to_numpy().astype(float)
    hist_pop = np.asarray(hist @ log_pop).ravel() / np.maximum(np.asarray(hist.sum(axis=1)).ravel(), 1)
    return pd.DataFrame({
        "case_id": part["case_id"].to_numpy(),
        "history_size": n_prior,
        "log_history": np.log1p(n_prior),
        "history_popularity": hist_pop,
    }), log_pop[final_items[:, :K]].mean(axis=1)


def signals_for(cfg, run, split, data, log_pop, scale):
    items_df, inter, cases = data
    d = log_dir(cfg, run, split)
    with np.load(d / "trajectory.npz") as z:
        items, scores = z["items"], z["scores"]
        dist, norm = z["dist_to_final"], z["final_norm"]
    case_log = pd.read_parquet(d / "cases.parquet")
    sig = sg.compute_signals(items, scores, dist, norm, scale, k=K)
    df = pd.DataFrame({"case_id": case_log["case_id"].to_numpy(),
                       "user_id": case_log["user_id"].to_numpy(), **sig})
    part = cases[cases["split"] == split].set_index("case_id").loc[df["case_id"]].reset_index()
    ctrl, rec_pop = controls(part, inter, len(items_df), log_pop, items[:, -1, :])
    df["rec_popularity"] = rec_pop
    df = df.merge(ctrl, on="case_id", how="left", validate="one_to_one")
    if df[["log_history", "history_popularity"]].isna().any().any():
        raise RuntimeError(f"{run} {split}: cases in the log missing from the processed data")
    df.to_parquet(d / "signals.parquet", index=False)
    return df


def single_auroc(df, labels):
    """AUROC of each signal alone. Labels joined here only, for the report split."""
    cols = [c for c in df.columns if c not in ("case_id", "user_id", "history_size")]
    risk_eval.check_predictors(cols)
    lab = labels[["case_id", "target_rank_final"]]
    m = df.merge(lab, on="case_id", how="inner", validate="one_to_one")
    if len(m) != len(df):
        raise RuntimeError("labels and signals do not cover the same cases")
    miss = (m["target_rank_final"] >= K).to_numpy()
    out = {"cases": len(m), "miss_rate": float(miss.mean())}
    for c in cols:
        out[c] = float(roc_auc_score(miss, m[c])) if m[c].nunique() > 1 else np.nan
    return out


def choose(table, prefix, options):
    """Option with the highest mean AUROC over runs (report split only)."""
    best = max(options, key=lambda o: table.loc[f"{prefix}_{o}", "mean"])
    return best, float(table.loc[f"{prefix}_{best}", "mean"])


def write_report(name, backbone, split, per_run, path):
    runs = list(per_run)
    tab = pd.DataFrame({r: per_run[r] for r in runs})
    meta = tab.loc[["cases", "miss_rate"]]
    tab = tab.drop(index=["cases", "miss_rate"])
    tab["mean"], tab["sd"] = tab[runs].mean(axis=1), tab[runs].std(axis=1, ddof=1)
    weights = [sg.weight_tag(r) for r in sg.RHOS]
    chosen = {
        "u_set_weighting": choose(tab, "u_set", weights),
        "u_rank_weighting": choose(tab, "u_rank", weights),
        "score_change_weighting": choose(tab, "score_change", weights),
        "entropy_temperature": choose(tab, "entropy", [f"t{t:g}" for t in sg.TAUS]),
    }
    lines = [f"# Section 5 signals: {name} ({backbone})", "",
             f"Single-signal AUROC on **{split}** (miss = target not in final top-{K}).",
             "Higher signal = higher risk; AUROC < 0.5 means the signal points the other way.",
             "Choices below are made on this split only; the test split is not used.", "",
             "Runs: " + ", ".join(f"{r} ({int(meta.loc['cases', r]):,} cases, "
                                   f"miss {meta.loc['miss_rate', r]:.3f})" for r in runs), ""]
    groups = dict(sg.SIGNAL_GROUPS, controls=sg.CONTROLS)
    lines += ["| Group | Signal | Mean AUROC | SD | " + " | ".join(runs) + " |",
              "|---|---|---|---|" + "---|" * len(runs)]
    for g, cols in groups.items():
        for c in cols:
            if c in tab.index:
                r = tab.loc[c]
                lines.append(f"| {g} | `{c}` | {r['mean']:.3f} | {r['sd']:.3f} | "
                             + " | ".join(f"{r[x]:.3f}" for x in runs) + " |")
    lines += ["", "## Choices on this split", ""]
    for k, (opt, auc) in chosen.items():
        lines.append(f"- {k}: `{opt}` (mean AUROC {auc:.3f})")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps(
        {"dataset": name, "backbone": backbone, "split": split,
         "chosen": {k: v[0] for k, v in chosen.items()},
         "auroc": {r: per_run[r] for r in runs}}, indent=2, default=float))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["valid_model", "valid_risk", "test"])
    parser.add_argument("--report-split", default="valid_risk", choices=["valid_model", "valid_risk"])
    args = parser.parse_args()
    cfg = load_cfg(args.config)

    data = dd.load_processed(cfg)
    train, _ = dd.train_matrix(data[1], len(data[0]))
    log_pop = np.log1p(item_counts(train))

    per_run, backbones = {}, set()
    for run in args.runs:
        manifest = json.loads((log_dir(cfg, run, args.splits[0]) / "manifest.json").read_text())
        backbones.add(manifest.get("backbone", "DiffRec"))
        scale = score_scale(cfg, run)
        for split in args.splits:
            t0 = time.time()
            df = signals_for(cfg, run, split, data, log_pop, scale)
            print(f"{run} {split}: {len(df):,} cases, {time.time() - t0:.0f} s", flush=True)
            if split == args.report_split:
                labels = pd.read_parquet(log_dir(cfg, run, split) / "labels.parquet")
                per_run[run] = single_auroc(df, labels)
    if len(backbones) != 1:
        raise SystemExit(f"Mix of backbones {backbones}: run DiffRec and GiffCF separately.")
    backbone = backbones.pop()
    if per_run:
        name = Path(args.config).stem
        rep = REPO / "reports" / "signals"
        rep.mkdir(parents=True, exist_ok=True)
        suffix = "" if backbone == "DiffRec" else "_giffcf"
        print(write_report(name, backbone, args.report_split, per_run, rep / f"{name}{suffix}"))


if __name__ == "__main__":
    main()
