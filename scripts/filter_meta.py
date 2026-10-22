"""Download an Amazon 2023 metadata file (if missing) and keep only our items.

The full meta_Books file has millions of books; we only need those in the
ratings file. The kept lines are written unchanged, so scripts/embed_items.py
reads the result exactly like meta_Video_Games.jsonl.gz.

    sbatch jobs/run.sh configs/amazon_books.yaml scripts/filter_meta.py --meta meta_Books.jsonl.gz --out meta_Books_kept.jsonl.gz
"""
import argparse
import gzip
import json
import shutil
import time
import urllib.request
from pathlib import Path

import pandas as pd

from trajrisk.config import load_config

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config()
    raw = Path(cfg["paths"].get("raw_dir", Path(cfg["paths"]["data_dir"]) / "raw"))
    print(f"Raw folder: {raw}")
    src = raw / args.meta

    if not src.exists():
        t0 = time.time()
        part = src.with_name(src.name + ".part")
        print(f"Downloading {BASE + args.meta} ...", flush=True)
        with urllib.request.urlopen(BASE + args.meta, timeout=120) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, length=16 * 1024 * 1024)
        part.rename(src)
        print(f"Downloaded {src.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f} s")
    else:
        print(f"Using existing {src} ({src.stat().st_size / 1e9:.2f} GB)")

    rf = raw / cfg["data"]["raw_file"]
    if rf.suffix == ".parquet":
        asins = pd.read_parquet(rf, columns=["parent_asin"])["parent_asin"]
    else:
        asins = pd.read_csv(rf, usecols=["parent_asin"])["parent_asin"]
    asins = set(asins.unique())
    print(f"{len(asins):,} items in the ratings file {rf.name}")

    t0, n, kept = time.time(), 0, 0
    with gzip.open(src, "rt") as fin, gzip.open(raw / args.out, "wt") as fout:
        for line in fin:
            n += 1
            if json.loads(line).get("parent_asin") in asins:
                fout.write(line)
                kept += 1
            if n % 1_000_000 == 0:
                print(f"  {n:,} lines read, {kept:,} kept", flush=True)
    print(f"Done: {n:,} lines, kept {kept:,} ({kept / len(asins):.1%} of our items have metadata), "
          f"{time.time() - t0:.0f} s -> {raw / args.out}")


if __name__ == "__main__":
    main()
