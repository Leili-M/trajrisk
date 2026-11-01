"""Proposal section 3, step 1: recipe attributes for the controlled benchmark.

1. Builds a clean attribute table for every HUMMUS catalog recipe
   ($data_dir/processed/benchmark/item_attributes.pkl), used by all later steps.
2. Audits the attributes before any scenario is built:
   - how many catalog recipes carry each hard-constraint value (tags, time)
   - whether the tags can be trusted: vegetarian / vegan tags against the
     ingredient lists, time tags against the recorded duration
   - which soft preferences conflict (negative rank correlation)
   - how many recipes satisfy each combination of hard constraints, to check that
     scenarios with many, few and no feasible recipes can be built

    sbatch --partition=gpumedium --time=00:30:00 jobs/run.sh configs/base.yaml \
        scripts/benchmark_attributes.py --config configs/base.yaml

Writes reports/benchmark/attributes.md and attributes.json in the repository.
"""

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from trajrisk import diffrec_data as dd
from trajrisk import recipes as rc

REPO = Path(__file__).resolve().parents[1]
SOFT = ["duration", "n_ingredients", "calories", "protein", "fsa_score"]
# Direction of each soft preference: -1 = smaller is preferred, +1 = larger is preferred
PREFERENCE = {"duration": ("shorter time", -1), "n_ingredients": ("fewer ingredients", -1),
              "calories": ("fewer calories", -1), "protein": ("more protein", +1),
              "fsa_score": ("healthier (FSA)", +1)}
BUCKETS = [(0, 0, "0"), (1, 9, "1-9"), (10, 50, "10-50"), (51, 200, "51-200"),
           (201, 10**9, ">200")]


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


def read_recipes(raw_dir):
    cols = ["new_recipe_id", "title", "tags", "ingredients"] + list(rc.NUMERIC_COLUMNS.values())
    head = pd.read_csv(raw_dir / "pp_recipes.zip", nrows=0).columns
    return pd.read_csv(raw_dir / "pp_recipes.zip", usecols=[c for c in cols if c in head],
                       low_memory=False)


def share(x):
    x = pd.Series(x).dropna().astype(bool)
    return float(x.mean()) if len(x) else float("nan")


def coverage(a):
    out = {"recipes": int(len(a)), "with_tags": int(a["has_tags"].sum()),
           "with_ingredients": int(a["ingredients"].notna().sum())}
    for c in rc.NUMERIC_COLUMNS:
        out[f"with_{c}"] = int(a[c].notna().sum())
    return out


def tag_counts(a):
    rows = {}
    for t in rc.DIET_TAGS + rc.COURSE_TAGS + rc.CUISINE_TAGS + list(rc.TIME_TAGS):
        rows[t] = int(a[f"tag:{t}"].fillna(False).astype(bool).sum())
    return rows


def tag_reliability(a):
    veg = a[a["tag:vegetarian"].fillna(False).astype(bool)]
    vegan = a[a["tag:vegan"].fillna(False).astype(bool)]
    untagged_veg = a[a["has_tags"] & ~a["tag:vegetarian"].fillna(False).astype(bool)]
    rel = {
        "vegetarian_tagged": int(len(veg)),
        "vegetarian_tagged_mentioning_meat": int(veg["mentions_meat"].fillna(False).sum()),
        "vegan_tagged": int(len(vegan)),
        "vegan_tagged_mentioning_animal_product": int(vegan["mentions_animal_product"].fillna(False).sum()),
        "not_tagged_vegetarian_but_no_meat_word": int((untagged_veg["mentions_meat"] == False).sum()),  # noqa: E712
        "examples_vegetarian_with_meat_word": veg.loc[veg["mentions_meat"].fillna(False).astype(bool),
                                                     ["title", "ingredients"]].head(15)
                                                 .to_dict("records"),
        "time_tags": {},
    }
    for t, limit in rc.TIME_TAGS.items():
        tagged = a[a[f"tag:{t}"].fillna(False).astype(bool) & a["duration"].notna()]
        rel["time_tags"][t] = {"tagged_with_duration": int(len(tagged)),
                               "duration_within_limit": share(tagged["duration"] <= limit)}
    return rel


def soft_correlations(a):
    """Spearman correlation between preference scores (oriented so higher = preferred).

    A negative value means the two preferences conflict: recipes that are good for
    one tend to be bad for the other.
    """
    oriented = pd.DataFrame({PREFERENCE[c][0]: PREFERENCE[c][1] * a[c].rank(pct=True) for c in SOFT})
    c = oriented.corr(method="pearson")                    # Pearson on ranks = Spearman
    names = list(oriented.columns)
    return {f"{x} vs {y}": float(c.loc[x, y]) for x, y in itertools.combinations(names, 2)}


def feasible_grid(a):
    """Number of recipes that certainly satisfy each combination of hard constraints.

    A recipe counts only if every required attribute is known and satisfied;
    an unknown attribute never counts as satisfied.
    """
    diet = [None] + rc.DIET_TAGS
    course = [None] + rc.COURSE_TAGS
    time = [None, 15, 30, 60]
    cuisine = [None] + rc.CUISINE_TAGS
    rows = []
    masks = {f"tag:{t}": a[f"tag:{t}"].fillna(False).astype(bool).to_numpy()
             for t in rc.DIET_TAGS + rc.COURSE_TAGS + rc.CUISINE_TAGS}
    dur = a["duration"].to_numpy()
    for d, c, t, k in itertools.product(diet, course, time, cuisine):
        m = np.ones(len(a), dtype=bool)
        if d: m &= masks[f"tag:{d}"]
        if c: m &= masks[f"tag:{c}"]
        if k: m &= masks[f"tag:{k}"]
        if t: m &= np.nan_to_num(dur, nan=np.inf) <= t
        n_constraints = sum(x is not None for x in (d, c, t, k))
        if n_constraints >= 2:
            rows.append({"diet": d, "course": c, "max_time": t, "cuisine": k,
                         "n_constraints": n_constraints, "feasible": int(m.sum())})
    g = pd.DataFrame(rows)
    hist = {lab: int(((g["feasible"] >= lo) & (g["feasible"] <= hi)).sum()) for lo, hi, lab in BUCKETS}
    return g, hist


def to_markdown(r):
    L = ["# Proposal section 3, step 1: recipe attributes of the HUMMUS catalog", ""]
    cv = r["coverage"]
    L += [f"Catalog recipes: {cv['recipes']:,}", "", "| Attribute | Recipes with a value | Share |",
          "|---|---|---|"]
    for k, v in cv.items():
        if k != "recipes":
            L.append(f"| {k.replace('with_', '')} | {v:,} | {100 * v / cv['recipes']:.1f}% |")
    L += ["", "## Hard-constraint values (recipes carrying the tag)", "",
          "| Tag | Recipes |", "|---|---|"]
    L += [f"| {k} | {v:,} |" for k, v in r["tag_counts"].items()]
    rel = r["tag_reliability"]
    L += ["", "## Can the tags be trusted?", "",
          f"- Tagged vegetarian: {rel['vegetarian_tagged']:,}; of these, ingredient list mentions "
          f"a meat or fish word: {rel['vegetarian_tagged_mentioning_meat']:,} "
          f"({100 * rel['vegetarian_tagged_mentioning_meat'] / max(rel['vegetarian_tagged'], 1):.1f}%)",
          f"- Tagged vegan: {rel['vegan_tagged']:,}; of these, mentions an animal product: "
          f"{rel['vegan_tagged_mentioning_animal_product']:,} "
          f"({100 * rel['vegan_tagged_mentioning_animal_product'] / max(rel['vegan_tagged'], 1):.1f}%)",
          f"- Recipes with tags but without the vegetarian tag whose ingredients mention no meat "
          f"word: {rel['not_tagged_vegetarian_but_no_meat_word']:,} (tags are incomplete, so "
          f"'not tagged' is not the same as 'not vegetarian')", "",
          "| Time tag | Tagged recipes with a duration | Duration within the limit |", "|---|---|---|"]
    for t, v in rel["time_tags"].items():
        L.append(f"| {t} | {v['tagged_with_duration']:,} | {100 * v['duration_within_limit']:.1f}% |")
    L += ["", "Examples of vegetarian-tagged recipes with a meat word (for manual inspection):", ""]
    L += [f"- {e['title']}: {str(e['ingredients'])[:160]}" for e in rel["examples_vegetarian_with_meat_word"]]
    L += ["", "## Soft preferences: rank correlation between attributes", "",
          "Preferences are oriented so that higher means preferred. A negative value means the "
          "two preferences conflict: recipes that are good for one tend to be bad for the other.", "", "| Pair | Spearman |", "|---|---|"]
    L += [f"| {k} | {v:+.2f} |" for k, v in r["soft_correlations"].items()]
    L += ["", "## Feasible recipes per combination of 2-4 hard constraints", "",
          "| Feasible recipes | Combinations |", "|---|---|"]
    L += [f"| {k} | {v:,} |" for k, v in r["feasible_histogram"].items()]
    L += ["", "Full grid: reports/benchmark/feasible_grid.csv", ""]
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    items, _, _ = dd.load_processed(cfg)

    attrs = rc.build_attributes(read_recipes(data_dir / "raw"), items)
    store = data_dir / "processed" / "benchmark"
    store.mkdir(parents=True, exist_ok=True)
    attrs.to_pickle(store / "item_attributes.pkl")

    grid, hist = feasible_grid(attrs)
    result = {"coverage": coverage(attrs), "tag_counts": tag_counts(attrs),
              "tag_reliability": tag_reliability(attrs),
              "soft_correlations": soft_correlations(attrs),
              "feasible_histogram": hist}
    out = REPO / "reports" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "attributes.json").write_text(json.dumps(result, indent=2, default=str))
    (out / "attributes.md").write_text(to_markdown(result))
    grid.to_csv(out / "feasible_grid.csv", index=False)
    print(to_markdown(result))
    print(f"\nAttribute table: {store / 'item_attributes.pkl'} ({len(attrs):,} recipes)")


if __name__ == "__main__":
    main()
