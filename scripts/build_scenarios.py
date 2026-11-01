"""Proposal section 3, step 2: structured scenario specifications.

Each scenario pairs one real HUMMUS evaluation case (user history up to the case
time) with a constructed request: hard constraints, soft-preference weights, the
feasible recipes, the reference (best feasible) list and the abstention outcome.
Validation scenarios use users from valid_risk, test scenarios users from test.

    sbatch --partition=gpumedium --time=00:30:00 jobs/run.sh configs/base.yaml \
        scripts/build_scenarios.py --config configs/base.yaml --n-valid 200 --n-test 400

Writes $data_dir/processed/benchmark/scenarios.jsonl and
reports/benchmark/scenarios.md / scenarios.json in the repository.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from trajrisk import benchmark as bm
from trajrisk import diffrec_data as dd

REPO = Path(__file__).resolve().parents[1]


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


def describe(s):
    h = ", ".join(f"{k}={v}" for k, v in s["hard"].items() if v is not None)
    w = ", ".join(f"{k} {v}" for k, v in s["soft"].items())
    return (f"{s['scenario_id']} ({s['feasible_type']}, {s['soft_type']}): hard [{h}]; "
            f"soft [{w}]; feasible {s['n_feasible']:,}, unknown {s['n_unknown']:,}")


def summary(scen, combos):
    df = pd.DataFrame(scen)
    out = {"scenarios": int(len(df)),
           "users": int(df["user_id"].nunique()),
           "by_split_and_type": df.groupby(["split", "feasible_type"]).size().unstack(fill_value=0)
                                  .to_dict("index"),
           "by_split_and_soft": df.groupby(["split", "soft_type"]).size().unstack(fill_value=0)
                                  .to_dict("index"),
           "heldout_in_validation": int(((df["split"] != "test")
                                         & (df["heldout_soft"] | df["heldout_course"])).sum()),
           "heldout_in_test": int(((df["split"] == "test")
                                   & (df["heldout_soft"] | df["heldout_course"])).sum()),
           "users_in_both_splits": int(len(set(df.loc[df["split"] == "test", "user_id"])
                                           & set(df.loc[df["split"] != "test", "user_id"]))),
           "feasible_size": df.groupby("feasible_type")["n_feasible"]
                              .describe()[["min", "50%", "max"]].to_dict("index"),
           "abstain_scenarios": int(df["abstain"].sum()),
           "hard_combinations_available": combos["feasible_type"].value_counts().to_dict()}
    return out


def to_markdown(s, examples):
    L = ["# Proposal section 3, step 2: scenario specifications", "",
         f"Scenarios: {s['scenarios']:,}, one per user ({s['users']:,} users; "
         f"users in both splits: {s['users_in_both_splits']}).", "",
         "| Split | " + " | ".join(bm.FEASIBLE_TYPES) + " |", "|---|" + "---|" * len(bm.FEASIBLE_TYPES)]
    for split, v in s["by_split_and_type"].items():
        L.append(f"| {split} | " + " | ".join(str(v.get(t, 0)) for t in bm.FEASIBLE_TYPES) + " |")
    L += ["", "| Split | single | aligned | conflicting |", "|---|---|---|---|"]
    for split, v in s["by_split_and_soft"].items():
        L.append(f"| {split} | {v.get('single', 0)} | {v.get('aligned', 0)} | {v.get('conflicting', 0)} |")
    L += ["", "Feasible recipes per scenario type (min / median / max):", ""]
    L += [f"- {t}: {v['min']:.0f} / {v['50%']:.0f} / {v['max']:.0f}" for t, v in s["feasible_size"].items()]
    L += ["", f"Scenarios with no feasible recipe (abstention expected): {s['abstain_scenarios']}",
          f"Held-out combinations in validation: {s['heldout_in_validation']} (must be 0); "
          f"in test: {s['heldout_in_test']}",
          f"Held-out preference pairs: {bm.HELDOUT_SOFT}; held-out course: {bm.HELDOUT_COURSE}",
          "", "Available hard-constraint combinations per type: "
          + ", ".join(f"{k} {v}" for k, v in s["hard_combinations_available"].items()),
          "", "## Examples", ""]
    L += [f"- {describe(e)}" for e in examples]
    return "\n".join(L) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n-valid", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=400)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    rng = np.random.default_rng(cfg["project"]["seed"])

    store = Path(cfg["paths"]["data_dir"]) / "processed" / "benchmark"
    attrs = pd.read_pickle(store / "item_attributes.pkl").sort_values("item_idx").reset_index(drop=True)
    pref = bm.preference_scores(attrs)
    _, _, cases = dd.load_processed(cfg)

    combos = bm.hard_combinations(attrs)
    val = bm.sample_scenarios(combos, cases[cases["split"] == "valid_risk"], args.n_valid,
                              "valid", rng)
    test = bm.sample_scenarios(combos, cases[cases["split"] == "test"], args.n_test, "test", rng,
                               exclude_users={s["user_id"] for s in val})
    specs = val + test
    scen = [bm.complete_scenario(s, attrs, pref) for s in specs]

    with open(store / "scenarios.jsonl", "w") as f:
        for s in scen:
            f.write(json.dumps(s) + "\n")
    s = summary(scen, combos)
    out = REPO / "reports" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "scenarios.json").write_text(json.dumps(s, indent=2, default=str))
    examples = [next(x for x in scen if x["feasible_type"] == t and x["soft_type"] == st)
                for t, st in [("many", "single"), ("few", "aligned"), ("many", "conflicting"),
                              ("none", "single"), ("few", "conflicting")]
                if any(x["feasible_type"] == t and x["soft_type"] == st for x in scen)]
    (out / "scenarios.md").write_text(to_markdown(s, examples))
    print(to_markdown(s, examples))
    print(f"Scenarios: {store / 'scenarios.jsonl'}")


if __name__ == "__main__":
    main()
