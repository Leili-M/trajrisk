"""Proposal section 7: calibration transfer to unseen users.

The risk model is fitted on the risk-validation cases (as in scripts/judge.py) and
applied once to the test cases. The test cases are then split by user: users who also
have risk-validation cases ("seen") and users who do not ("unseen"). If the risk model
transfers, AUROC, Brier score and calibration-in-the-large (mean predicted miss
probability against the observed miss rate) should be similar in both groups.
Nothing is chosen on the test cases; the predictor sets and the section-5 choices are
the same as in scripts/judge.py.

    sbatch --partition=gpumedium --time=00:30:00 jobs/run.sh configs/amazon_video_games.yaml \
        scripts/transfer.py --config configs/amazon_video_games.yaml \
        --runs vg_base vg_seed1 vg_seed2 vg_seed3 vg_seed4
Report: reports/transfer/<dataset>[_giffcf].md and .json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge  # noqa: E402  (same loaders, predictor sets and fitting as the main analysis)
from trajrisk import judge_eval as je  # noqa: E402

SETS = ("controls", "controls + lists", "combined")
GROUPS = ("seen", "unseen")


def group_scores(prob, miss):
    miss = np.asarray(miss, dtype=bool)
    out = {"cases": int(len(miss)), "miss_rate": float(miss.mean()),
           "mean_predicted": float(np.mean(prob)),
           "brier": float(brier_score_loss(miss, np.clip(prob, 0, 1)))}
    out["auroc"] = float(roc_auc_score(miss, prob)) if 0 < miss.sum() < len(miss) else float("nan")
    out["calibration_gap"] = out["mean_predicted"] - out["miss_rate"]
    return out


def evaluate_run(cfg, run, sets):
    fit = judge.load_split(cfg, run, judge.FIT_SPLIT)
    test = judge.load_split(cfg, run, judge.EVAL_SPLIT)
    seen_users = set(fit["user_id"].unique())
    is_seen = test["user_id"].isin(seen_users).to_numpy()
    res = {"share_unseen_cases": float((~is_seen).mean()), "sets": {}}
    for name in SETS:
        prob, _ = je.fit_predict(fit, test, sets[name])
        res["sets"][name] = {
            "seen": group_scores(prob[is_seen], test["miss"].to_numpy()[is_seen]),
            "unseen": group_scores(prob[~is_seen], test["miss"].to_numpy()[~is_seen]),
        }
    return res


def write_report(name, backbone, per_run, path):
    runs = list(per_run)
    share = je.t_interval([per_run[r]["share_unseen_cases"] for r in runs])[0]
    lines = [f"# Section 7: calibration transfer to unseen users, {name} ({backbone})", "",
             f"Risk model fitted on {judge.FIT_SPLIT}, applied once to {judge.EVAL_SPLIT}. "
             "Seen: test users who also have risk-validation cases; unseen: all other test "
             f"users. Share of test cases from unseen users: {share:.3f}. "
             "Mean over runs with 95% t-interval.", "",
             "| Predictors | Users | Cases | Miss rate | Mean predicted | Calibration gap | "
             "Brier | AUROC |", "|---|---|---|---|---|---|---|---|"]
    f = lambda v: "–" if np.isnan(v[0]) else (f"{v[0]:.3f}" if np.isnan(v[1])
                                             else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")
    summary = {}
    for name_set in SETS:
        for g in GROUPS:
            vals = {m: je.t_interval([per_run[r]["sets"][name_set][g][m] for r in runs])
                    for m in ("cases", "miss_rate", "mean_predicted", "calibration_gap",
                              "brier", "auroc")}
            summary[f"{name_set} | {g}"] = {m: list(v) for m, v in vals.items()}
            lines.append(f"| `{name_set}` | {g} | {vals['cases'][0]:,.0f} | "
                         f"{vals['miss_rate'][0]:.3f} | {vals['mean_predicted'][0]:.3f} | "
                         f"{f(vals['calibration_gap'])} | {f(vals['brier'])} | "
                         f"{f(vals['auroc'])} |")
    lines += ["", "Calibration gap: mean predicted miss probability minus observed miss rate "
              "(0 = calibrated in the large)."]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps(
        {"dataset": name, "backbone": backbone, "summary": summary, "runs": per_run},
        indent=2, default=float))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    cfg = judge.load_cfg(args.config)
    name = Path(args.config).stem
    log = Path(cfg["paths"]["output_dir"]) / "trajlogs"
    backbones = {json.loads((log / f"{r}__{judge.EVAL_SPLIT}" / "manifest.json").read_text())
                 .get("backbone", "DiffRec") for r in args.runs}
    if len(backbones) != 1:
        raise SystemExit(f"Mix of backbones {backbones}: run DiffRec and GiffCF separately.")
    backbone = backbones.pop()
    suffix = "" if backbone == "DiffRec" else "_giffcf"
    chosen = json.loads((judge.REPO / "reports" / "signals" / f"{name}{suffix}.json").read_text())
    sets, _ = judge.predictor_sets(chosen["chosen"])
    per_run = {run: evaluate_run(cfg, run, sets) for run in args.runs}
    out = judge.REPO / "reports" / "transfer"
    out.mkdir(parents=True, exist_ok=True)
    print(write_report(name, backbone, per_run, out / f"{name}{suffix}"))


if __name__ == "__main__":
    main()
