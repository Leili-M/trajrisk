"""Proposal section 9, step 2: uncertainty-guided fallback to EASE.

For every run: cases with predicted risk above a threshold get the EASE list, the others
keep the diffusion list. The threshold is chosen on risk-validation (cross-fitted judge
predictions, folds split by user) to maximise NDCG@10, and applied once to test.

Policies (risk score used for routing):
  judge: controls + lists    trajectory-aware judge (list signals, no score change)
  judge: controls            final-output / history judge (no trajectory)
  entropy                    untrained final-score confidence
Compared with: always diffusion, always EASE, random routing at the same rate
(expected value), and an outcome-based oracle (diagnostic only).

Reported on test: NDCG@10 and Recall@10 over ALL cases, routing rate, gain on routed
cases, catalog coverage, mean log-popularity and tail share of recommended items,
routing rate by history length, and cost per case.

    sbatch --partition=gpumedium --time=01:00:00 jobs/run.sh configs/amazon_books.yaml \
        scripts/routing.py --config configs/amazon_books.yaml \
        --runs bk_seed42 bk_seed1 bk_seed2 bk_seed3 bk_seed4
Report: reports/routing/<dataset>[_giffcf].md, .json and curve .csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge  # noqa: E402
from subgroups import HIST_BINS, history_group, item_counts  # noqa: E402

from trajrisk import diffrec_data as dd  # noqa: E402
from trajrisk import judge_eval as je  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RATES = np.round(np.arange(0, 1.0001, 0.05), 2)


def quality(routed, d, f):
    return float(np.where(routed, f, d).mean())


def route_by_rate(risk, rate):
    """Route the `rate` share of cases with the highest risk."""
    n = int(round(rate * len(risk)))
    routed = np.zeros(len(risk), dtype=bool)
    if n:
        routed[np.argsort(-risk, kind="stable")[:n]] = True
    return routed


def choose_threshold(risk_val, ndcg_d, ndcg_f):
    """Risk threshold that maximises validation NDCG@10 (route if risk > threshold)."""
    best, best_q = np.inf, quality(np.zeros(len(risk_val), bool), ndcg_d, ndcg_f)
    for r in RATES[1:]:
        routed = route_by_rate(risk_val, r)
        q = quality(routed, ndcg_d, ndcg_f)
        if q > best_q:
            best_q = q
            best = float(np.min(risk_val[routed])) - 1e-12
    return best


def list_stats(lists, counts, tail_cut):
    rec = lists.ravel()
    return {"coverage": len(np.unique(rec)) / len(counts),
            "mean_log_pop": float(np.log1p(counts[rec]).mean()),
            "tail_share": float((counts[rec] <= tail_cut).mean())}


def load_case_data(cfg, run, split, fb_dir):
    df = judge.load_split(cfg, run, split)
    fb = pd.read_parquet(fb_dir / f"ease__{split}" / "ranks.parquet")
    df = df.merge(fb, on="case_id", how="inner", validate="one_to_one")
    miss_f, loss_f = je.labels_from_ranks(df.pop("fallback_rank"))
    df["ndcg_d"], df["ndcg_f"] = 1 - df["loss_ndcg"], 1 - loss_f
    df["hit_d"], df["hit_f"] = (~df["miss"]).astype(float), (~miss_f).astype(float)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    cfg = judge.load_cfg(args.config)
    name = Path(args.config).stem
    out_root = Path(cfg["paths"]["output_dir"])
    log, fb_dir = out_root / "trajlogs", out_root / "fallback"

    backbone = json.loads((log / f"{args.runs[0]}__test" / "manifest.json").read_text()) \
        .get("backbone", "DiffRec")
    suffix = "" if backbone == "DiffRec" else "_giffcf"
    chosen = json.loads((REPO / "reports" / "signals" / f"{name}{suffix}.json").read_text())
    sets, _ = judge.predictor_sets(chosen["chosen"])
    entropy_col = f"entropy_{chosen['chosen']['entropy_temperature']}"

    items, inter, _ = dd.load_processed(cfg)
    train, _ = dd.train_matrix(inter, len(items))
    counts = item_counts(train)
    tail_cut = float(np.quantile(counts, 0.5))
    with np.load(fb_dir / "ease__test" / "top10.npz") as z:     # row order of ranks.parquet
        fb_lists = pd.DataFrame(z["top10"], index=pd.read_parquet(
            fb_dir / "ease__test" / "ranks.parquet")["case_id"])
    fb_cost = json.loads((fb_dir / "ease__test" / "manifest.json").read_text())

    per_run, curves = {}, []
    for run in args.runs:
        val = load_case_data(cfg, run, "valid_risk", fb_dir)
        test = load_case_data(cfg, run, "test", fb_dir)
        risks = {
            "judge: controls + lists": (
                je.cv_predict(val, sets["controls + lists"], val["user_id"].to_numpy()),
                je.fit_predict(val, test, sets["controls + lists"])[0]),
            "judge: controls": (
                je.cv_predict(val, sets["controls"], val["user_id"].to_numpy()),
                je.fit_predict(val, test, sets["controls"])[0]),
            "entropy": (val[entropy_col].to_numpy(), test[entropy_col].to_numpy()),
        }
        d, f = test["ndcg_d"].to_numpy(), test["ndcg_f"].to_numpy()
        hd, hf = test["hit_d"].to_numpy(), test["hit_f"].to_numpy()
        tdir = log / f"{run}__test"
        with np.load(tdir / "trajectory.npz") as z:
            d_lists = pd.DataFrame(z["items"][:, -1, :10],
                                   index=pd.read_parquet(tdir / "cases.parquet")["case_id"])
        d_lists = d_lists.loc[test["case_id"]].to_numpy()
        f_lists = fb_lists.loc[test["case_id"]].to_numpy()
        d_cost = json.loads((tdir / "manifest.json").read_text())["runtime_seconds"]["ms_per_case"]
        groups = history_group(test["history_size"].to_numpy())

        res = {"always diffusion": {"ndcg@10": float(d.mean()), "recall@10": float(hd.mean()),
                                    "rate": 0.0, **list_stats(d_lists, counts, tail_cut),
                                    "ms_per_case": d_cost},
               "always EASE": {"ndcg@10": float(f.mean()), "recall@10": float(hf.mean()),
                               "rate": 1.0, **list_stats(f_lists, counts, tail_cut),
                               "ms_per_case": fb_cost["ms_per_case_mean"]}}
        for pol, (rv, rt) in risks.items():
            thr = choose_threshold(rv, val["ndcg_d"].to_numpy(), val["ndcg_f"].to_numpy())
            routed = rt > thr
            rate = float(routed.mean())
            mixed = np.where(routed[:, None], f_lists, d_lists)
            res[pol] = {
                "ndcg@10": quality(routed, d, f), "recall@10": quality(routed, hd, hf),
                "rate": rate,
                "gain_on_routed_ndcg": float((f[routed] - d[routed]).mean()) if routed.any() else 0.0,
                "random_same_rate_ndcg": float((1 - rate) * d.mean() + rate * f.mean()),
                **list_stats(mixed, counts, tail_cut),
                "ms_per_case": d_cost + rate * fb_cost["ms_per_case_mean"],
                "rate_by_history": {g: float(routed[groups == g].mean())
                                    for g in pd.unique(groups)},
            }
            for r in RATES:
                curves.append({"run": run, "policy": pol, "rate": r,
                               "ndcg@10": quality(route_by_rate(rt, r), d, f)})
        gain = np.maximum(f - d, 0)
        for r in RATES:
            curves.append({"run": run, "policy": "random (expected)", "rate": r,
                           "ndcg@10": float((1 - r) * d.mean() + r * f.mean())})
            curves.append({"run": run, "policy": "oracle (diagnostic)", "rate": r,
                           "ndcg@10": quality(route_by_rate(gain + 1e-9 * (gain > 0), r)
                                              & (gain > 0), d, f)})
        per_run[run] = res
        print(f"{run}: " + ", ".join(f"{k} {v['ndcg@10']:.4f} (rate {v['rate']:.2f})"
                                     for k, v in res.items()), flush=True)

    # report: mean over runs with t-interval
    pols = list(next(iter(per_run.values())))
    lines = [f"# Section 9: routing to EASE, {name} ({backbone})", "",
             "Threshold chosen on risk-validation (cross-fitted judge), applied once to test. "
             f"EASE lambda {fb_cost['lambda']:g} (section 4). Mean over runs "
             "[95% t-interval]. Quality is over ALL test cases.", "",
             "| Policy | NDCG@10 | Recall@10 | Routed | Gain on routed (NDCG) | Random, same rate "
             "| Coverage | Mean log-pop | Tail share | ms/case |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    f3 = lambda m: "{:.4f} [{:.4f}, {:.4f}]".format(*je.t_interval(m))
    for p in pols:
        g = lambda k: [per_run[r][p].get(k, np.nan) for r in per_run]
        lines.append(f"| {p} | {f3(g('ndcg@10'))} | {np.mean(g('recall@10')):.4f} | "
                     f"{np.mean(g('rate')):.2f} | "
                     + (f"{np.mean(g('gain_on_routed_ndcg')):+.4f}" if "gain_on_routed_ndcg"
                        in per_run[args.runs[0]][p] else "–") + " | "
                     + (f"{np.mean(g('random_same_rate_ndcg')):.4f}" if "random_same_rate_ndcg"
                        in per_run[args.runs[0]][p] else "–")
                     + f" | {np.mean(g('coverage')):.3f} | {np.mean(g('mean_log_pop')):.2f} | "
                     f"{np.mean(g('tail_share')):.3f} | {np.mean(g('ms_per_case')):.3f} |")
    lines += ["", "## Routing rate by history length (test, mean over runs)", "",
              "| Policy | " + " | ".join(g[2] for g in HIST_BINS) + " |",
              "|---|" + "---|" * 5]
    for p in pols:
        if "rate_by_history" in per_run[args.runs[0]][p]:
            vals = [np.nanmean([per_run[r][p]["rate_by_history"].get(h[2], np.nan)
                                for r in per_run]) for h in HIST_BINS]
            lines.append(f"| {p} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    lines += ["", f"Fallback cost: build {fb_cost['build_seconds']} s, "
              f"{fb_cost['ms_per_case_mean']} ms/case (p95 batch {fb_cost['ms_per_case_p95_batch']}), "
              f"peak GPU memory {fb_cost['peak_gpu_memory_gb']} GB. Diffusion cost per case is the "
              "logged trajectory run (includes logging, an upper bound). The routed cost always "
              "includes the full diffusion run, because the risk needs the completed trajectory.",
              "", "Quality-rate curves: `<dataset>_curve.csv` (policies, random, oracle)."]
    out = REPO / "reports" / "routing"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}{suffix}.md").write_text("\n".join(lines) + "\n")
    pd.DataFrame(curves).to_csv(out / f"{name}{suffix}_curve.csv", index=False)
    (out / f"{name}{suffix}.json").write_text(json.dumps(
        {"runs": per_run, "fallback": fb_cost, "tail_cut": tail_cut}, indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
