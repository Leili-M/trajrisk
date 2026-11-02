"""Proposal section 8: results by history length and target popularity, and paired
confidence intervals by resampling users.

The judge is fitted once on all risk-validation cases (as in scripts/judge.py) and its
test predictions are evaluated inside subgroups. Subgroup boundaries are fixed before
looking at results:
  history length (earlier positives): 1, 2-3, 4-9, 10-29, 30+
  target popularity: positive training interactions of the target item, split at the
  50% and 90% quantiles of the catalog items' training counts (tail / mid / head).
The target and its popularity are used ONLY to form evaluation groups, never as input.

Paired user bootstrap (per run): users of the test split are resampled with replacement
(B times); the same weights are used for both predictor sets, so the interval is for the
AUROC gain.

    sbatch --partition=gpumedium --time=02:00:00 jobs/run.sh configs/amazon_books.yaml \
        scripts/subgroups.py --config configs/amazon_books.yaml \
        --runs bk_seed42 bk_seed1 bk_seed2 bk_seed3 bk_seed4
Report: reports/subgroups/<dataset>[_giffcf].md and .json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge  # noqa: E402  (same predictor sets and data loading as the main judge)

from trajrisk import diffrec_data as dd  # noqa: E402
from trajrisk import judge_eval as je  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HIST_BINS = [(1, 1, "1"), (2, 3, "2-3"), (4, 9, "4-9"), (10, 29, "10-29"), (30, np.inf, "30+")]
COMPARISONS = [("controls3", "controls3 + lists"), ("controls", "controls + lists"),
               ("controls", "combined")]


def item_counts(train):
    try:
        import torch
        if torch.is_tensor(train):
            t = train.coalesce() if train.is_sparse else train
            s = torch.sparse.sum(t, dim=0).to_dense() if t.is_sparse else t.sum(dim=0)
            return s.cpu().numpy().astype(float).ravel()
    except ImportError:
        pass
    return np.asarray(train.sum(axis=0), dtype=float).ravel()


def history_group(n):
    out = np.empty(len(n), dtype=object)
    for lo, hi, name in HIST_BINS:
        out[(n >= lo) & (n <= hi)] = name
    return out


def popularity_group(target_counts, catalog_counts):
    q50, q90 = np.quantile(catalog_counts, [0.5, 0.9])
    return np.where(target_counts > q90, "head", np.where(target_counts > q50, "mid", "tail")), \
        (float(q50), float(q90))


def safe_auc(y, p, w=None):
    return roc_auc_score(y, p, sample_weight=w) if len(np.unique(y)) == 2 else np.nan


def paired_bootstrap(miss, users, preds, b, seed):
    """95% interval of AUROC(top) - AUROC(base) with users resampled with replacement."""
    rng = np.random.default_rng(seed)
    codes, n_users = pd.factorize(users)[0], len(np.unique(users))
    out = {k: [] for k in preds}
    for _ in range(b):
        w = rng.multinomial(n_users, np.full(n_users, 1 / n_users))[codes].astype(float)
        keep = w > 0
        for (base, top) in preds:
            out[(base, top)].append(safe_auc(miss[keep], preds[(base, top)][1][keep], w[keep])
                                    - safe_auc(miss[keep], preds[(base, top)][0][keep], w[keep]))
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    cfg = judge.load_cfg(args.config)
    name = Path(args.config).stem
    log = Path(cfg["paths"]["output_dir"]) / "trajlogs"

    backbone = json.loads((log / f"{args.runs[0]}__test" / "manifest.json").read_text()) \
        .get("backbone", "DiffRec")
    suffix = "" if backbone == "DiffRec" else "_giffcf"
    chosen = json.loads((REPO / "reports" / "signals" / f"{name}{suffix}.json").read_text())
    sets, _ = judge.predictor_sets(chosen["chosen"])

    items, inter, _ = dd.load_processed(cfg)
    train, _ = dd.train_matrix(inter, len(items))
    counts = item_counts(train)

    rows, boot = [], {}
    for run in args.runs:
        fit, test = judge.load_split(cfg, run, judge.FIT_SPLIT), judge.load_split(cfg, run, "test")
        lab = pd.read_parquet(log / f"{run}__test" / "labels.parquet")[["case_id", "target_idx"]]
        test = test.merge(lab, on="case_id", how="left", validate="one_to_one")
        test["hist_group"] = history_group(test["history_size"].to_numpy())
        test["pop_group"], bounds = popularity_group(counts[test["target_idx"].to_numpy()], counts)
        prob = {s: je.fit_predict(fit, test, sets[s])[0]
                for s in {x for pair in COMPARISONS for x in pair}}
        miss = test["miss"].to_numpy()
        for kind, col in (("history", "hist_group"), ("target popularity", "pop_group")):
            for g, idx in test.groupby(col).indices.items():
                r = {"run": run, "kind": kind, "group": g, "cases": len(idx),
                     "share": len(idx) / len(test), "miss_rate": float(miss[idx].mean())}
                for base, top in COMPARISONS:
                    a0, a1 = safe_auc(miss[idx], prob[base][idx]), safe_auc(miss[idx], prob[top][idx])
                    r[f"auroc {base}"], r[f"gain {top} over {base}"] = a0, a1 - a0
                rows.append(r)
        boot[run] = paired_bootstrap(miss, test["user_id"].to_numpy(),
                                     {c: (prob[c[0]], prob[c[1]]) for c in COMPARISONS},
                                     args.bootstrap, cfg["project"]["seed"])
        print(f"{run}: done", flush=True)

    df = pd.DataFrame(rows)
    order = {g[2]: i for i, g in enumerate(HIST_BINS)} | {"tail": 0, "mid": 1, "head": 2}
    num = [c for c in df.columns if c not in ("run", "kind", "group")]
    agg = df.groupby(["kind", "group"])[num].mean().reset_index()
    agg["o"] = agg["group"].map(order)
    agg = agg.sort_values(["kind", "o"])
    lines = [f"# Section 8: subgroups and user bootstrap, {name} ({backbone})", "",
             f"Judge fitted on {judge.FIT_SPLIT}, evaluated on test. Mean over runs: "
             + ", ".join(args.runs) + ".",
             f"Target popularity: tail <= {bounds[0]:g} < mid <= {bounds[1]:g} < head "
             "(training positives; 50% / 90% quantiles of catalog items).", "",
             "| Kind | Group | Share of cases | Miss rate | AUROC controls3 | "
             + " | ".join(f"Gain {t} over {b}" for b, t in COMPARISONS) + " |",
             "|---|---|---|---|---|" + "---|" * len(COMPARISONS)]
    for _, r in agg.iterrows():
        lines.append(f"| {r['kind']} | {r['group']} | {r['share']:.3f} | {r['miss_rate']:.3f} | "
                     f"{r['auroc controls3']:.3f} | "
                     + " | ".join(f"{r[f'gain {t} over {b}']:+.3f}" for b, t in COMPARISONS) + " |")
    lines += ["", f"## Paired user bootstrap ({args.bootstrap} resamples), 95% interval of the gain", "",
              "| Run | " + " | ".join(f"{t} over {b}" for b, t in COMPARISONS) + " |",
              "|---|" + "---|" * len(COMPARISONS)]
    for run, d in boot.items():
        lines.append(f"| {run} | " + " | ".join(f"[{d[c][0]:+.3f}, {d[c][1]:+.3f}]"
                                               for c in COMPARISONS) + " |")
    out = REPO / "reports" / "subgroups"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}{suffix}.md").write_text("\n".join(lines) + "\n")
    (out / f"{name}{suffix}.json").write_text(json.dumps(
        {"rows": rows, "bootstrap": {r: {f"{t} over {b}": v for (b, t), v in d.items()}
                                     for r, d in boot.items()},
         "popularity_bounds": bounds, "history_bins": [g[2] for g in HIST_BINS]},
        indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
