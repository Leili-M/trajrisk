"""Score the simple methods on one split, overall and by history length.

    TRAJRISK_CONFIG=configs/movielens.yaml PYTHONPATH=src python scripts/run_baselines.py

Results go to <output_dir>/baselines/<split>.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajrisk import diffrec_data as dd
from trajrisk import evaluate as ev
from trajrisk.baselines import all_methods
from trajrisk.config import load_config

# Groups by number of earlier liked items (same as Part 6 of the report)
HISTORY_BINS = [0, 1, 4, 19, 99, np.inf]
HISTORY_LABELS = ["1", "2-4", "5-19", "20-99", "100+"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="valid_model")
    parser.add_argument("--recent-days", type=int, default=365)
    parser.add_argument("--only", nargs="*", help="run only these methods")
    args = parser.parse_args()

    cfg = load_config()
    ks = cfg["evaluation"]["ks"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items, inter, cases = dd.load_processed(cfg)
    n_items = len(items)
    train, _ = dd.train_matrix(inter, n_items)
    part = cases[cases["split"] == args.split].reset_index(drop=True)
    hist, seen = dd.case_matrices(part, inter, n_items)
    targets = part["target_idx"].to_numpy()
    groups = pd.cut(part["n_prior"], HISTORY_BINS, labels=HISTORY_LABELS)

    print(f"{cfg['data']['dataset']}: {len(part):,} {args.split} cases, "
          f"{n_items:,} items, device {device}")

    methods = all_methods(train, inter, n_items, device, args.recent_days)
    if args.only:
        methods = {k: v for k, v in methods.items() if k in args.only}

    rows = []
    for name, fn in methods.items():
        t0 = time.time()
        ranks = np.asarray(ev.rank_cases(fn, hist, seen, targets, device))
        overall = ev.metrics_from_ranks(ranks, ks)
        rows.append({"method": name, "group": "all", "cases": len(ranks), **overall})
        for g in HISTORY_LABELS:
            mask = (groups == g).to_numpy()
            if mask.any():
                rows.append({"method": name, "group": g, "cases": int(mask.sum()),
                             **ev.metrics_from_ranks(ranks[mask], ks)})
        print(f"  {name:18s} recall@10={overall['recall@10']:.4f}  "
              f"ndcg@10={overall['ndcg@10']:.4f}  ({time.time() - t0:.0f} s)")

    df = pd.DataFrame(rows)
    out_dir = Path(cfg["paths"]["output_dir"]) / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{args.split}.csv", index=False)

    labels = ["all"] + [g for g in HISTORY_LABELS if g in set(df["group"])]
    table = df.pivot(index="method", columns="group", values="recall@10")
    print("\nRecall@10 by number of earlier liked items:")
    print(table.reindex(index=list(methods), columns=labels).round(4).to_string())
    print(f"\nSaved to {out_dir / f'{args.split}.csv'}")


if __name__ == "__main__":
    main()