"""Proposal section 3, step 5: human audit of a sample of scenarios.

make:       draws a stratified random sample of scenarios and writes a review sheet
            (CSV, opens in Excel) with the request texts and three example feasible
            recipes; the reviewer fills five 1/0 columns following docs/audit_guide.md.
summarize:  reads the filled sheet and reports the share of scenarios that pass
            each check, and compares the reviewer's judgement of the paraphrases
            with the automatic parser results of step 4.

    PYTHONPATH=src python scripts/audit_sheet.py make --config configs/base.yaml --n 100
    (fill reports/benchmark/audit/audit_sheet.csv, save it as audit_sheet_filled.csv)
    PYTHONPATH=src python scripts/audit_sheet.py summarize --config configs/base.yaml
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports" / "benchmark" / "audit"
CHECKS = {"clear": "The request is clear and unambiguous",
          "feasible_ok": "The example recipes really satisfy every hard constraint "
                         "(empty for scenarios with no feasible recipe)",
          "attributes_ok": "The recipe data contain what is needed to judge the request",
          "para1_same": "Paraphrase 1 keeps every requirement and the preference order",
          "para2_same": "Paraphrase 2 keeps every requirement and the preference order"}


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


def stratified_sample(scenarios, n, seed):
    """Equal share from every (feasible type, preference type) cell, then fill at random."""
    df = pd.DataFrame(scenarios)
    cells = list(df.groupby(["feasible_type", "soft_type"]))
    per_cell = max(1, n // len(cells))
    picked = [g.sample(min(per_cell, len(g)), random_state=seed) for _, g in cells]
    picked = pd.concat(picked)
    rest = df[~df["scenario_id"].isin(picked["scenario_id"])]
    extra = rest.sample(n=min(len(rest), max(0, n - len(picked))), random_state=seed)
    return sorted(pd.concat([picked, extra])["scenario_id"].tolist())[:n]


def _readable_hard(h):
    return "; ".join(f"{k}={v}" for k, v in h.items() if v is not None)


def _readable_soft(s):
    return "; ".join(f"{k} {v}" for k, v in sorted(s.items(), key=lambda kv: -kv[1]))


def _recipe_line(row, hard):
    tags = [t for t in ("vegetarian", "vegan", hard.get("course"), hard.get("cuisine"))
            if t and row.get(f"tag:{t}") is True]
    dur = "?" if pd.isna(row["duration"]) else f"{row['duration']:.0f} min"
    ing = str(row["ingredients"])[:140]
    return f"{row['title']} | {dur} | tags: {', '.join(tags) or '-'} | ingredients: {ing}"


def make(cfg, n):
    store = Path(cfg["paths"]["data_dir"]) / "processed" / "benchmark"
    scen = {s["scenario_id"]: s for s in map(json.loads, open(store / "scenarios.jsonl"))}
    attrs = pd.read_pickle(store / "item_attributes.pkl").set_index("item_idx")
    texts = pd.DataFrame(map(json.loads, open(store / "requests_llm.jsonl")))
    rng = np.random.default_rng(cfg["project"]["seed"])
    ids = stratified_sample(list(scen.values()), n, cfg["project"]["seed"])
    rows = []
    for sid in ids:
        s = scen[sid]
        t = texts[texts["scenario_id"] == sid]
        para = t[t["source"] == "llm"].sort_values("variant")["text"].tolist() + ["", ""]
        examples = rng.choice(s["feasible"], size=min(3, len(s["feasible"])), replace=False) \
            if s["feasible"] else []
        row = {"scenario_id": sid, "feasible_type": s["feasible_type"], "soft_type": s["soft_type"],
               "hard": _readable_hard(s["hard"]), "soft": _readable_soft(s["soft"]),
               "n_feasible": s["n_feasible"],
               "template_text": t[(t["source"] == "template") & (t["variant"] == 0)]["text"].iloc[0],
               "paraphrase_1": para[0], "paraphrase_2": para[1]}
        for k in range(3):
            row[f"example_{k + 1}"] = _recipe_line(attrs.loc[examples[k]], s["hard"]) \
                if k < len(examples) else ""
        row.update({c: "" for c in CHECKS})
        row["notes"] = ""
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "audit_sheet.csv", index=False, encoding="utf-8-sig")
    print(f"{len(rows)} scenarios -> {OUT / 'audit_sheet.csv'}")


def summarize(cfg):
    store = Path(cfg["paths"]["data_dir"]) / "processed" / "benchmark"
    sheet = pd.read_csv(OUT / "audit_sheet_filled.csv", encoding="utf-8-sig")
    texts = pd.DataFrame(map(json.loads, open(store / "requests_llm.jsonl")))
    res = {"reviewed": int(len(sheet))}
    for c in CHECKS:
        v = pd.to_numeric(sheet[c], errors="coerce").dropna()
        res[c] = {"judged": int(len(v)), "pass_share": float(v.mean()) if len(v) else float("nan")}
    # Reviewer's view of each paraphrase against the automatic parser of step 4
    llm = texts[texts["source"] == "llm"].set_index(["scenario_id", "variant"])
    pairs = []
    for _, r in sheet.iterrows():
        for v, col in ((0, "para1_same"), (1, "para2_same")):
            same = pd.to_numeric(pd.Series([r[col]]), errors="coerce").iloc[0]
            if pd.notna(same) and (r["scenario_id"], v) in llm.index:
                auto = llm.loc[(r["scenario_id"], v)]
                pairs.append({"same_meaning": int(same), "llm_parser_correct": bool(auto["llm_all_correct"]),
                              "rule_parser_correct": bool(auto["rule_all_correct"])})
    p = pd.DataFrame(pairs)
    if len(p):
        res["paraphrases_judged"] = int(len(p))
        res["paraphrases_meaning_kept"] = float(p["same_meaning"].mean())
        kept = p[p["same_meaning"] == 1]
        res["llm_parser_correct_when_meaning_kept"] = float(kept["llm_parser_correct"].mean()) if len(kept) else None
        res["rule_parser_correct_when_meaning_kept"] = float(kept["rule_parser_correct"].mean()) if len(kept) else None
        res["table_meaning_vs_llm_parser"] = pd.crosstab(p["same_meaning"], p["llm_parser_correct"]).to_dict()
    (OUT / "audit_summary.json").write_text(json.dumps(res, indent=2))
    L = ["# Proposal section 3, step 5: human audit", "",
         f"Reviewed scenarios: {res['reviewed']} (stratified random sample; one reviewer).", "",
         "| Check | Judged | Pass |", "|---|---|---|"]
    L += [f"| {CHECKS[c]} | {res[c]['judged']} | {100 * res[c]['pass_share']:.1f}% |" for c in CHECKS]
    if len(p):
        L += ["", f"Paraphrases judged: {res['paraphrases_judged']}; meaning kept: "
                  f"{100 * res['paraphrases_meaning_kept']:.1f}%.",
              f"Among paraphrases that kept their meaning, the language-model parser was correct for "
              f"{100 * res['llm_parser_correct_when_meaning_kept']:.1f}% and the rule-based parser for "
              f"{100 * res['rule_parser_correct_when_meaning_kept']:.1f}%."]
    notes = sheet["notes"].dropna().astype(str)
    notes = notes[notes.str.strip() != ""]
    if len(notes):
        L += ["", "## Reviewer notes", ""] + [f"- {sid}: {n}" for sid, n in
                                              zip(sheet.loc[notes.index, "scenario_id"], notes)]
    (OUT / "audit_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["make", "summarize"])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    make(cfg, args.n) if args.action == "make" else summarize(cfg)


if __name__ == "__main__":
    main()
