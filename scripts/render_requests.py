"""Proposal section 3, step 3: natural-language requests from the scenario specifications.

Every scenario is rendered with three sentence templates. The rule-based parser
then reads each text back; on template text it must recover every constraint and
the order of the preferences (a consistency check of the rendering). The same
parser is applied to free paraphrases in step 4.

    sbatch --partition=gpumedium --time=00:15:00 jobs/run.sh configs/base.yaml \
        scripts/render_requests.py --config configs/base.yaml

Writes $data_dir/processed/benchmark/requests.jsonl (one row per scenario and
template) and reports/benchmark/requests.md in the repository.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from trajrisk import request_text as rt

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = (0, 1, 2)


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    store = Path(cfg["paths"]["data_dir"]) / "processed" / "benchmark"
    scen = [json.loads(line) for line in open(store / "scenarios.jsonl")]

    rows = []
    for s in scen:
        for t in TEMPLATES:
            text = rt.render(s, t)
            parsed = rt.parse(text)
            rows.append({"scenario_id": s["scenario_id"], "split": s["split"], "source": "template",
                         "variant": t, "text": text, "parsed": parsed,
                         **rt.compare(parsed, s)})
    with open(store / "requests.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    df = pd.DataFrame(rows)
    acc = df.groupby("variant")[["hard_exact", "soft_set", "soft_order", "all_correct"]].mean()
    L = ["# Proposal section 3, step 3: natural-language requests", "",
         f"Scenarios: {len(scen):,}; texts: {len(df):,} (three templates per scenario).", "",
         "Parser accuracy on template text (consistency check, should be 100%):", "",
         "| Template | Hard constraints exact | Preference set | Preference order | All correct |",
         "|---|---|---|---|---|"]
    for v, r in acc.iterrows():
        L.append(f"| {v} | {100 * r.hard_exact:.1f}% | {100 * r.soft_set:.1f}% | "
                 f"{100 * r.soft_order:.1f}% | {100 * r.all_correct:.1f}% |")
    L += ["", "## Examples", ""]
    for s in scen[:4]:
        L.append(f"**{s['scenario_id']}** hard {s['hard']}, soft {s['soft']}")
        L += [f"- {rt.render(s, t)}" for t in TEMPLATES]
        L.append("")
    bad = df[~df["all_correct"]].head(10)
    if len(bad):
        L += ["## Parser errors (first 10)", ""] + [f"- {r.text} -> {r.parsed}" for r in bad.itertuples()]
    out = REPO / "reports" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "requests.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
