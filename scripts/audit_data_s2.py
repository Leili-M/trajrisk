"""Proposal section 2: audit of every dataset before the experiment is frozen.

For each dataset this script reports
  A  source files: size, SHA-256 and the download manifest (version record)
  B  duplicates: identical rows and users who rated the same item more than once
  C  rating scale, dropped ratings and the share of positive interactions
  D  filters applied by the data provider (minimum events per user and item)
  E  the frozen split: dates, share of events and share of time in each part,
     and how many test events could be evaluated with a split by time period instead
  F  attrition: how many events and users remain after every rule
  G  attribute coverage of the catalog items (needed for the controlled benchmark)

    sbatch --partition=gpumedium --time=02:00:00 jobs/run.sh configs/movielens.yaml \
        scripts/audit_data_s2.py --configs configs/movielens.yaml configs/base.yaml \
        configs/amazon_video_games.yaml configs/amazon_books.yaml

Writes reports/checks/proposal_s2.md and reports/checks/proposal_s2.json in the repository.
"""

import argparse
import gzip
import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from trajrisk import diffrec_data as dd
from trajrisk.data import LOADERS, add_split

REPO = Path(__file__).resolve().parents[1]
SPLITS = ["train", "valid_model", "valid_risk", "test"]
EVAL_SPLITS = SPLITS[1:]
# Amazon metadata fields that describe the item. average_rating and rating_number are
# left out on purpose: they summarise reviews up to the crawl date, i.e. the future.
AMAZON_FIELDS = ["title", "description", "features", "categories", "main_category",
                 "details", "store", "price", "images", "videos"]


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


# ------------------------------------------------------------------ A. source files
def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def source_files(raw_dir):
    files = {}
    for p in sorted(raw_dir.iterdir()):
        if p.is_file() and p.name != "wget.log":
            files[p.name] = {"bytes": p.stat().st_size,
                             "sha256": sha256(p) if p.name != "download_manifest.json" else None}
    manifest = raw_dir / "download_manifest.json"
    return {"files": files,
            "manifest": json.loads(manifest.read_text()) if manifest.exists() else None}


# ------------------------------------------------------------------ B-D. raw interactions
def duplicates(df):
    """Identical rows, and (user, item) pairs that occur more than once."""
    pairs = df.groupby(["user_id", "item_id"]).size()
    return {
        "identical_rows": int(df.duplicated(["user_id", "item_id", "rating", "date"]).sum()),
        "same_user_item_same_time": int(df.duplicated(["user_id", "item_id", "date"]).sum()),
        "user_item_pairs_seen_more_than_once": int((pairs > 1).sum()),
        "extra_rows_from_repeated_pairs": int((pairs - 1).sum()),
    }


def ratings(raw, kept, positive_ratings):
    return {
        "rating_counts_raw": {str(k): int(v) for k, v in raw["rating"].value_counts().sort_index().items()},
        "rows_dropped_by_rating_rule": int(len(raw) - len(kept)),
        "positive_ratings": [float(r) for r in positive_ratings],
        "positive_share": float(kept["rating"].isin(positive_ratings).mean()),
    }


def provider_filters(df):
    """Minimum events per user and item in the file as delivered (e.g. 5-core)."""
    u, i = df.groupby("user_id").size(), df.groupby("item_id").size()
    return {"events": int(len(df)), "users": int(len(u)), "items": int(len(i)),
            "min_events_per_user": int(u.min()), "min_events_per_item": int(i.min()),
            "first_event": str(df["date"].min()), "last_event": str(df["date"].max())}


# ------------------------------------------------------------------ E. split
def eligible_after(df, positive, start):
    """Positive events after `start` whose user had a positive event strictly earlier.

    Same rule as the evaluation cases, before the catalog filter, so that two
    choices of split can be compared on equal terms.
    """
    pos = df[positive]
    first = pos.groupby("user_id")["date"].min()
    later = pos[pos["date"] > start]
    ok = later["date"].to_numpy() > first.reindex(later["user_id"]).to_numpy()
    return {"events": int(ok.sum()), "users": int(later.loc[ok, "user_id"].nunique())}


def split_table(df, cfg, positive):
    s = cfg["data"]["split"]
    t0, t1 = df["date"].min(), df["date"].max()
    bounds = [t0, pd.Timestamp(s["train_end"]), pd.Timestamp(s["valid_model_end"]),
              pd.Timestamp(s["valid_end"]), t1]
    span = (t1 - t0).total_seconds()
    labelled = add_split(df, cfg)
    parts = {}
    for k, name in enumerate(SPLITS):
        part = labelled[labelled["split"] == name]
        parts[name] = {"from": str(bounds[k]), "to": str(bounds[k + 1]),
                       "events": int(len(part)), "event_share": float(len(part) / len(df)),
                       "time_share": float((bounds[k + 1] - bounds[k]).total_seconds() / span),
                       "users": int(part["user_id"].nunique())}
    time_test_start = t0 + 0.8 * (t1 - t0)          # proposal: last 20% of the time period
    return {
        "method": s.get("method", "event_quantile"),
        "parts": parts,
        "test_eligible_event_split": eligible_after(df, positive, bounds[3]),
        "time_period_split": {"train_end": str(t0 + 0.7 * (t1 - t0)),
                              "test_start": str(time_test_start),
                              "test_eligible": eligible_after(df, positive, time_test_start)},
    }


# ------------------------------------------------------------------ F. attrition
def attrition(raw, kept, inter, cases, cfg):
    """Events and users left after each rule, per evaluation split."""
    positive_ratings = cfg["data"]["positive_ratings"]
    raw_s, kept_s = add_split(raw, cfg), add_split(kept, cfg)
    first = inter.groupby(["user_id", "item_idx"])["date"].min().rename("first_date").reset_index()
    pos_cat = inter[inter["positive"]].merge(first, on=["user_id", "item_idx"])
    pos_cat["repeat"] = pos_cat["date"] > pos_cat["first_date"]
    case_rep = cases.merge(first, left_on=["user_id", "target_idx"],
                           right_on=["user_id", "item_idx"])
    out = {}
    for s in EVAL_SPLITS:
        r, k = raw_s[raw_s["split"] == s], kept_s[kept_s["split"] == s]
        kp = k[k["rating"].isin(positive_ratings)]
        pc = pos_cat[pos_cat["split"] == s]
        c = cases[cases["split"] == s]
        steps = [("all events", r), ("after rating rule", k), ("positive", kp),
                 ("positive on catalog items", pc), ("with an earlier positive (cases)", c)]
        out[s] = {
            "steps": [{"step": n, "events": int(len(d)), "users": int(d["user_id"].nunique())}
                      for n, d in steps],
            "positive_on_catalog_repeating_an_earlier_interaction": int(pc["repeat"].sum()),
            "cases_repeating_an_earlier_interaction":
                int((case_rep.loc[case_rep["split"] == s, "date"]
                     > case_rep.loc[case_rep["split"] == s, "first_date"]).sum()),
        }
    return out


# ------------------------------------------------------------------ G. attributes
def nonempty(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    if isinstance(v, str):
        return v.strip() not in ("", "None", "nan", "[]")
    if isinstance(v, (list, dict, tuple)):
        return len(v) > 0
    return True


def coverage_hummus(raw_dir, kept_ids):
    counts, n, cols = None, 0, None
    for chunk in pd.read_csv(raw_dir / "pp_recipes.zip", chunksize=50_000, low_memory=False):
        if cols is None:
            cols = list(chunk.columns)
            id_col = "new_recipe_id" if "new_recipe_id" in cols else \
                next(c for c in cols if "recipe_id" in c)
        part = chunk[chunk[id_col].isin(kept_ids)]
        nz = part.apply(lambda col: col.map(nonempty)).sum()
        counts = nz if counts is None else counts + nz
        n += len(part)
    return {"id_column": id_col, "items_found": int(n),
            "share_with_value": {c: float(counts[c] / max(n, 1)) for c in cols if c != id_col}}


def coverage_movielens(raw_dir, kept_ids):
    with zipfile.ZipFile(raw_dir / "ml-1m.zip") as z, z.open("ml-1m/movies.dat") as f:
        m = pd.read_csv(f, sep="::", engine="python", header=None, encoding="latin-1",
                        names=["item_id", "title", "genres"])
    m = m[m["item_id"].isin(kept_ids)]
    return {"id_column": "item_id", "items_found": int(len(m)),
            "share_with_value": {"title": float(m["title"].map(nonempty).mean()),
                                 "genres": float((m["genres"].map(nonempty)
                                                  & (m["genres"] != "(no genres listed)")).mean())}}


def coverage_amazon(raw_dir, data_cfg, kept_ids):
    # Item numbers are the codes of the sorted parent_asin values (data._load_amazon)
    asins = pd.read_csv(raw_dir / data_cfg["raw_file"], usecols=["parent_asin"], dtype=str)
    categories = pd.Categorical(asins["parent_asin"]).categories
    kept = set(categories[np.asarray(sorted(kept_ids))])
    stem = data_cfg["raw_file"].split(".")[0]
    meta = raw_dir / f"meta_{stem}_kept.jsonl.gz"
    if not meta.exists():
        meta = raw_dir / f"meta_{stem}.jsonl.gz"
    counts, found = dict.fromkeys(AMAZON_FIELDS, 0), set()
    with gzip.open(meta, "rt") as f:
        for line in f:
            rec = json.loads(line)
            a = rec.get("parent_asin")
            if a in kept and a not in found:
                found.add(a)
                for k in AMAZON_FIELDS:
                    counts[k] += nonempty(rec.get(k))
    return {"metadata_file": meta.name, "items_found": len(found),
            "share_with_value": {k: v / len(kept) for k, v in counts.items()},
            "note": "average_rating and rating_number are not used (they include future reviews)"}


def attributes(cfg, raw_dir, items):
    kept_ids = set(items["item_id"].tolist())
    name = cfg["data"]["dataset"]
    if name == "hummus":
        res = coverage_hummus(raw_dir, kept_ids)
    elif name == "movielens":
        res = coverage_movielens(raw_dir, kept_ids)
    else:
        res = coverage_amazon(raw_dir, cfg["data"], kept_ids)
    res["catalog_items"] = len(kept_ids)
    return res


# ------------------------------------------------------------------ report
def pct(x):
    return f"{100 * x:.1f}%"


def to_markdown(results):
    L = ["# Proposal section 2: dataset audit", ""]
    L += ["| Dataset | Events | Users | Items | Min events/user | Min events/item | "
          "Identical rows | Repeated user-item pairs | Positive share |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        p, d = r["provider_filters"], r["duplicates"]
        L.append(f"| {n} | {p['events']:,} | {p['users']:,} | {p['items']:,} | "
                 f"{p['min_events_per_user']} | {p['min_events_per_item']} | "
                 f"{d['identical_rows']:,} | {d['user_item_pairs_seen_more_than_once']:,} | "
                 f"{pct(r['ratings']['positive_share'])} |")
    L.append("")
    for n, r in results.items():
        L += [f"## {n}", "", "### Source files", "", "| File | Bytes | SHA-256 |", "|---|---|---|"]
        for f, v in r["sources"]["files"].items():
            L.append(f"| {f} | {v['bytes']:,} | {v['sha256'] or ''} |")
        if r["sources"]["manifest"]:
            L += ["", "Download manifest:", "", "```json",
                  json.dumps(r["sources"]["manifest"], indent=2), "```"]
        L += ["", "### Ratings and duplicates", "",
              f"Raw rating counts: {r['ratings']['rating_counts_raw']}  ",
              f"Rows dropped by the rating rule: {r['ratings']['rows_dropped_by_rating_rule']:,}  ",
              f"Positive ratings: {r['ratings']['positive_ratings']}, "
              f"share {pct(r['ratings']['positive_share'])}  ",
              f"Duplicates: {r['duplicates']}  ",
              f"Period: {r['provider_filters']['first_event']} to {r['provider_filters']['last_event']}",
              "", "### Frozen split", "",
              "| Part | From | To | Events | Share of events | Share of time | Users |",
              "|---|---|---|---|---|---|---|"]
        for s, v in r["split"]["parts"].items():
            L.append(f"| {s} | {v['from']} | {v['to']} | {v['events']:,} | {pct(v['event_share'])} | "
                     f"{pct(v['time_share'])} | {v['users']:,} |")
        e, t = r["split"]["test_eligible_event_split"], r["split"]["time_period_split"]
        L += ["", f"Test events with an earlier positive (before catalog filter): "
                  f"event split {e['events']:,} events / {e['users']:,} users; "
                  f"time-period split (test from {t['test_start']}) "
                  f"{t['test_eligible']['events']:,} events / {t['test_eligible']['users']:,} users.",
              "", "### Attrition", ""]
        for s, v in r["attrition"].items():
            L += [f"**{s}**", "", "| Step | Events | Users |", "|---|---|---|"]
            L += [f"| {x['step']} | {x['events']:,} | {x['users']:,} |" for x in v["steps"]]
            L += ["", f"Positive catalog events repeating an earlier interaction: "
                      f"{v['positive_on_catalog_repeating_an_earlier_interaction']:,}; "
                      f"cases repeating an earlier interaction: "
                      f"{v['cases_repeating_an_earlier_interaction']:,}", ""]
        a = r["attributes"]
        L += ["### Attribute coverage of catalog items", "",
              f"Catalog items: {a['catalog_items']:,}, found in metadata: {a['items_found']:,}", "",
              "| Attribute | Share with a value |", "|---|---|"]
        L += [f"| {k} | {pct(v)} |" for k, v in a["share_with_value"].items()]
        if "note" in a:
            L += ["", a["note"]]
        L.append("")
    return "\n".join(L)


def audit(path):
    cfg = load_cfg(path)
    raw_dir = Path(cfg["paths"]["data_dir"]) / "raw"
    raw = LOADERS[cfg["data"]["dataset"]](raw_dir, cfg["data"])[["user_id", "item_id", "rating", "date"]]
    kept = raw[~raw["rating"].isin(cfg["data"]["drop_ratings"])]
    pr = cfg["data"]["positive_ratings"]
    items, inter, cases = dd.load_processed(cfg)
    return {
        "sources": source_files(raw_dir),
        "duplicates": duplicates(kept),
        "ratings": ratings(raw, kept, pr),
        "provider_filters": provider_filters(raw),
        "split": split_table(kept, cfg, kept["rating"].isin(pr)),
        "attrition": attrition(raw, kept, inter, cases, cfg),
        "attributes": attributes(cfg, raw_dir, items),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    args = parser.parse_args()
    results = {}
    for path in args.configs:
        name = Path(path).stem
        print(f"== {name}", flush=True)
        results[name] = audit(path)
        r = results[name]
        print(json.dumps({k: r[k] for k in ("duplicates", "ratings", "provider_filters")},
                         indent=1, default=str), flush=True)
    out = REPO / "reports" / "checks"
    out.mkdir(parents=True, exist_ok=True)
    (out / "proposal_s2.json").write_text(json.dumps(results, indent=2, default=str))
    (out / "proposal_s2.md").write_text(to_markdown(results))
    print(f"\nDone -> {out / 'proposal_s2.md'}")


if __name__ == "__main__":
    main()
