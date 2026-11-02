"""Proposal sections 7-8: risk model ("judge") and its evaluation.

Reads, for every run, the signals of scripts/compute_signals.py and the separate label
files. The judge (logistic regression) is fitted on risk-validation cases and applied
once to the test cases. Weightings and temperatures come from the section-5 choices
made on risk-validation (reports/signals/<dataset>[_giffcf].json).

Predictor sets (proposal section 7):
  controls3     log_history, small_margin, history_popularity (earlier criterion)
  controls      controls3 + rec_popularity + top_margin + entropy (final output / history)
  trajectory    trajectory signals only
  combined      controls + trajectory
Primary endpoint: AUROC gain of `combined` over `controls3` on test, mean over seeds
with a 95% t-interval (success if >= 0.02). Untrained single signals are reported too.

    sbatch --partition=gpumedium --time=01:00:00 jobs/run.sh configs/amazon_video_games.yaml \
        scripts/judge.py --config configs/amazon_video_games.yaml \
        --runs vg_base vg_seed1 vg_seed2 vg_seed3 vg_seed4
Report: reports/judge/<dataset>[_giffcf].md and .json, reliability tables in .csv
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from trajrisk import judge_eval as je

REPO = Path(__file__).resolve().parents[1]
FIT_SPLIT, EVAL_SPLIT = "valid_risk", "test"


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


def load_split(cfg, run, split):
    d = Path(cfg["paths"]["output_dir"]) / "trajlogs" / f"{run}__{split}"
    sig = pd.read_parquet(d / "signals.parquet")
    lab = pd.read_parquet(d / "labels.parquet")[["case_id", "target_rank_final"]]
    df = sig.merge(lab, on="case_id", how="inner", validate="one_to_one")
    if len(df) != len(sig):
        raise RuntimeError(f"{run} {split}: labels do not cover all cases")
    df["miss"], df["loss_ndcg"] = je.labels_from_ranks(df.pop("target_rank_final"))
    return df


def predictor_sets(chosen):
    u_set, u_rank = f"u_set_{chosen['u_set_weighting']}", f"u_rank_{chosen['u_rank_weighting']}"
    score = f"score_change_{chosen['score_change_weighting']}"
    entropy = f"entropy_{chosen['entropy_temperature']}"
    controls3 = ["log_history", "small_margin", "history_popularity"]
    controls = controls3 + ["rec_popularity", "top_margin", entropy]
    trajectory = [u_set, u_rank, score, "score_change_first", "cons_set", "cons_rank",
                  "reversal_set"]
    trajectory = list(dict.fromkeys(trajectory))
    return {"controls3": controls3, "controls": controls, "trajectory": trajectory,
            "controls3 + trajectory": controls3 + trajectory,
            "combined": controls + trajectory}, [u_set, u_rank, score, "score_change_first",
                                                  "cons_set", "small_margin"]


def evaluate_run(cfg, run, sets, singles, out_dir):
    fit, test = load_split(cfg, run, FIT_SPLIT), load_split(cfg, run, EVAL_SPLIT)
    res = {"cases_fit": len(fit), "cases_test": len(test),
           "miss_rate_test": float(test["miss"].mean()),
           "reference_aurc_miss": je.reference_aurc(test["miss"]),
           "reference_aurc_ndcg": je.reference_aurc(test["loss_ndcg"]), "sets": {}, "single": {},
           "cv_in_test": {}}
    for name, cols in sets.items():
        prob, _ = je.fit_predict(fit, test, cols)
        res["sets"][name] = je.scores(prob, test["miss"], test["loss_ndcg"])
        if name == "combined":
            je.reliability_table(prob, test["miss"]).to_csv(out_dir / f"reliability_{run}.csv",
                                                            index=False)
    for name in ("controls3", "controls3 + trajectory"):     # earlier protocol, for comparison
        prob = je.cv_predict(test, sets[name], test["user_id"].to_numpy())
        res["cv_in_test"][name] = je.scores(prob, test["miss"], test["loss_ndcg"])["auroc"]
    for c in singles:                                         # untrained scores
        s = je.scores(test[c].to_numpy(), test["miss"], test["loss_ndcg"])
        res["single"][c] = {k: s[k] for k in ("auroc", "auprc", "aurc_miss", "aurc_ndcg")}
    return res


def summarise(per_run, sets):
    runs = list(per_run)
    rows = []
    for name in sets:
        row = {"predictors": name}
        for m in ("auroc", "auprc", "brier", "aurc_miss", "aurc_ndcg"):
            mean, lo, hi = je.t_interval([per_run[r]["sets"][name][m] for r in runs])
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = mean, lo, hi
        rows.append(row)
    gains = {}
    for base, top in (("controls3", "combined"), ("controls3", "controls3 + trajectory"),
                      ("controls", "combined")):
        g = [per_run[r]["sets"][top]["auroc"] - per_run[r]["sets"][base]["auroc"] for r in runs]
        gains[f"{top} over {base}"] = je.t_interval(g)
    cv = [per_run[r]["cv_in_test"]["controls3 + trajectory"] - per_run[r]["cv_in_test"]["controls3"]
          for r in runs]
    gains["CV inside test (earlier protocol): controls3 + trajectory over controls3"] = je.t_interval(cv)
    return pd.DataFrame(rows), gains


def write_report(name, backbone, per_run, sets, path):
    table, gains = summarise(per_run, sets)
    runs = list(per_run)
    r0 = per_run[runs[0]]
    f = lambda m, lo, hi: f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
    lines = [f"# Sections 7-8: judge on {name} ({backbone})", "",
             f"Judge fitted on **{FIT_SPLIT}**, evaluated once on **{EVAL_SPLIT}**. "
             "Mean over runs with 95% t-interval.",
             "Runs: " + ", ".join(f"{r} (test {per_run[r]['cases_test']:,} cases, "
                                   f"miss {per_run[r]['miss_rate_test']:.3f})" for r in runs), "",
             "## Primary endpoint: AUROC gain (success if >= 0.02)", "",
             "| Comparison | Gain [95% CI] |", "|---|---|"]
    for k, (m, lo, hi) in gains.items():
        lines.append(f"| {k} | {f(m, lo, hi)} |")
    lines += ["", "## Predictor sets (test)", "",
              "| Predictors | AUROC | AUPRC | Brier | AURC (miss) | AURC (1-NDCG) |",
              "|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['predictors']} | " + " | ".join(
            f(r[m], r[f'{m}_lo'], r[f'{m}_hi'])
            for m in ("auroc", "auprc", "brier", "aurc_miss", "aurc_ndcg")) + " |")
    prev = np.mean([per_run[r]["miss_rate_test"] for r in runs])
    rand_m = np.mean([per_run[r]["reference_aurc_miss"]["random"] for r in runs])
    orac_m = np.mean([per_run[r]["reference_aurc_miss"]["oracle"] for r in runs])
    rand_n = np.mean([per_run[r]["reference_aurc_ndcg"]["random"] for r in runs])
    orac_n = np.mean([per_run[r]["reference_aurc_ndcg"]["oracle"] for r in runs])
    lines += ["", f"Failure prevalence (AUPRC of a random score): {prev:.3f}. "
              f"References (not deployable): AURC miss random {rand_m:.3f} / oracle {orac_m:.3f}; "
              f"AURC 1-NDCG random {rand_n:.3f} / oracle {orac_n:.3f}.", "",
              "## Untrained single signals (test, mean over runs)", "",
              "| Signal | AUROC | AUPRC | AURC (miss) | AURC (1-NDCG) |", "|---|---|---|---|---|"]
    for c in r0["single"]:
        vals = [np.mean([per_run[r]["single"][c][m] for r in runs])
                for m in ("auroc", "auprc", "aurc_miss", "aurc_ndcg")]
        lines.append(f"| `{c}` | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    lines += ["", "Predictor sets: " + "; ".join(f"**{k}** = {', '.join(v)}" for k, v in sets.items()),
              "", "Reliability tables of the combined judge: `reliability_<run>.csv`."]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps(
        {"dataset": name, "backbone": backbone, "fit_split": FIT_SPLIT, "eval_split": EVAL_SPLIT,
         "sets": sets, "gains": {k: list(v) for k, v in gains.items()}, "runs": per_run},
        indent=2, default=float))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    name = Path(args.config).stem

    log = Path(cfg["paths"]["output_dir"]) / "trajlogs"
    backbones = {json.loads((log / f"{r}__{EVAL_SPLIT}" / "manifest.json").read_text())
                 .get("backbone", "DiffRec") for r in args.runs}
    if len(backbones) != 1:
        raise SystemExit(f"Mix of backbones {backbones}: run DiffRec and GiffCF separately.")
    backbone = backbones.pop()
    suffix = "" if backbone == "DiffRec" else "_giffcf"
    chosen = json.loads((REPO / "reports" / "signals" / f"{name}{suffix}.json").read_text())
    if chosen["split"] != FIT_SPLIT:
        raise SystemExit("Section-5 choices must come from the risk-validation split.")
    sets, singles = predictor_sets(chosen["chosen"])

    out = REPO / "reports" / "judge"
    out.mkdir(parents=True, exist_ok=True)
    rel_dir = out / f"{name}{suffix}_reliability"
    rel_dir.mkdir(exist_ok=True)
    per_run = {}
    for run in args.runs:
        per_run[run] = evaluate_run(cfg, run, sets, singles, rel_dir)
        c = per_run[run]["sets"]
        print(f"{run}: AUROC controls3 {c['controls3']['auroc']:.3f}  "
              f"combined {c['combined']['auroc']:.3f}", flush=True)
    print(write_report(name, backbone, per_run, sets, out / f"{name}{suffix}"))


if __name__ == "__main__":
    main()
