"""RQ4: is the uncertainty signal useful in practice?

Two views, for one trained model:
1. Selective recommendation: show DiffRec's list only for the share of cases with the
   lowest predicted risk. How often is the shown list right?
2. Routing: for the riskiest cases, use a simple method's list instead. Overall Recall@10?

Risk = out-of-fold probability of a wrong list from: controls only, controls + trajectory,
controls + masking, all; plus random order and an oracle.

    sbatch jobs/run.sh configs/amazon_books.yaml scripts/routing_experiment.py \
        --traj TRAJ_SIGNALS.csv --mask MASK_SIGNALS.parquet --alt itemknn --name bk_seed42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajrisk import diffrec_data as dd
from trajrisk import evaluate as ev
from trajrisk import risk_eval
from trajrisk.baselines import all_methods
from trajrisk.config import load_config

CONTROLS = ["log_history", "small_margin", "history_popularity"]
TRAJ = ["step_change@10", "step_change@50", "step_score_change"]
MASK = ["mask_change@10", "mask_change@50", "mask_score_change", "mask_unstable@10"]
SHARES = np.round(np.arange(0, 1.0001, 0.1), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--alt", default="itemknn", help="simple method used for routing")
    parser.add_argument("--name", required=True, help="label for the output file")
    parser.add_argument("--split", default="valid_model")
    args = parser.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items, inter, cases = dd.load_processed(cfg)
    n_items = len(items)
    train, _ = dd.train_matrix(inter, n_items)
    part = cases[cases["split"] == args.split].reset_index(drop=True)
    hist, seen = dd.case_matrices(part, inter, n_items)
    users = part["user_id"].to_numpy()

    # Signals of the judged DiffRec list (same case order as `part`)
    traj = pd.read_csv(args.traj)
    assert (traj["case_id"].to_numpy() == part["case_id"].to_numpy()).all()
    mask = pd.read_parquet(args.mask)
    assert len(mask) == len(part)
    df = pd.DataFrame({c: traj[c] for c in TRAJ})
    for c in MASK + ["small_margin"]:
        df[c] = mask[c].to_numpy()
    df["miss"] = mask["miss"].to_numpy() > 0.5
    n_hist = np.asarray(hist.sum(axis=1)).ravel()
    log_pop = np.log1p(np.asarray(train.sum(axis=0)).ravel())
    df["log_history"] = np.log1p(n_hist)
    df["history_popularity"] = np.asarray(hist @ log_pop).ravel() / np.maximum(n_hist, 1)

    # Hits of DiffRec and of the simple method
    diff_hit = ~df["miss"].to_numpy()
    alt_fn = all_methods(train, inter, n_items, device)[args.alt]
    alt_ranks = np.asarray(ev.rank_cases(alt_fn, hist, seen, part["target_idx"].to_numpy(), device))
    alt_hit = alt_ranks < 10

    rng = np.random.default_rng(0)
    risks = {
        "random": rng.random(len(df)),
        "controls": risk_eval.cv_predict(df, CONTROLS, users),
        "controls+trajectory": risk_eval.cv_predict(df, CONTROLS + TRAJ, users),
        "controls+masking": risk_eval.cv_predict(df, CONTROLS + risk_eval.usable(df, MASK), users),
        "controls+both": risk_eval.cv_predict(df, CONTROLS + TRAJ + risk_eval.usable(df, MASK), users),
        "oracle": (~diff_hit).astype(float) + 0.5 * rng.random(len(df)),
    }

    rows = []
    for name, risk in risks.items():
        order = np.argsort(risk)          # most confident first
        for share in SHARES:
            # 1) selective: keep the `share` most confident cases
            k = int(round(share * len(df)))
            kept = order[:k]
            selective = diff_hit[kept].mean() if k else np.nan
            # 2) routing: send the riskiest `share` of cases to the simple method
            routed = np.zeros(len(df), dtype=bool)
            routed[order[len(df) - k:]] = True
            routing = np.where(routed, alt_hit, diff_hit).mean()
            rows.append({"name": args.name, "risk": name, "share": share,
                         "hit_rate_of_shown": selective, "recall_with_routing": routing})
    res = pd.DataFrame(rows)

    print(f"{args.name}: {len(df):,} cases | DiffRec Recall@10 {diff_hit.mean():.4f} | "
          f"{args.alt} Recall@10 {alt_hit.mean():.4f}")
    print("\n1) Hit rate of the shown lists when showing only the most confident share:")
    print(res.pivot(index="share", columns="risk", values="hit_rate_of_shown")
             .loc[[0.1, 0.2, 0.5, 0.8, 1.0]].round(4).to_string())
    print(f"\n2) Overall Recall@10 when the riskiest share is sent to {args.alt}:")
    print(res.pivot(index="share", columns="risk", values="recall_with_routing")
             .loc[[0.0, 0.2, 0.5, 0.8, 1.0]].round(4).to_string())

    out = Path(cfg["paths"]["output_dir"]) / "routing"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"{args.name}_{args.split}.csv", index=False)
    print("Saved to", out / f"{args.name}_{args.split}.csv")


if __name__ == "__main__":
    main()