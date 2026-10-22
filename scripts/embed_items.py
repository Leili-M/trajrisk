"""Embed the text of every kept item with a small sentence model (RQ2, text part).

    sbatch jobs/run.sh configs/amazon_video_games.yaml scripts/embed_items.py --meta meta_Video_Games.jsonl.gz

Saves <data_dir>/processed/item_text.parquet (item_idx, text)
  and <data_dir>/processed/item_emb.npy (n_items x 384, unit length).
Only the text is used (title + description); rating fields in the metadata use all
years and would leak future information, so they are ignored.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from trajrisk import diffrec_data as dd
from trajrisk.config import load_config

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def amazon_texts(raw_dir, data_cfg, meta_file, item_ids):
    """Title + description for each item, in item_idx order ('' if missing)."""
    ratings = pd.read_csv(raw_dir / data_cfg["raw_file"], usecols=["parent_asin"], dtype=str)
    asins = pd.Categorical(ratings["parent_asin"]).categories  # same numbering as data._load_amazon
    wanted = {asins[i]: pos for pos, i in enumerate(item_ids)}
    texts = [""] * len(item_ids)
    with gzip.open(raw_dir / meta_file, "rt") as f:
        for line in f:
            d = json.loads(line)
            pos = wanted.get(d.get("parent_asin"))
            if pos is not None:
                desc = " ".join(d.get("description") or [])
                texts[pos] = f"{d.get('title') or ''}. {desc}"[:1000].strip(" .")
    return texts


@torch.no_grad()
def embed(texts, device, batch_size=256):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    out = []
    for i in range(0, len(texts), batch_size):
        x = tok(texts[i:i + batch_size], padding=True, truncation=True,
                max_length=256, return_tensors="pt").to(device)
        h = model(**x).last_hidden_state
        m = x["attention_mask"].unsqueeze(-1)
        out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="metadata file in <data_dir>/raw")
    args = parser.parse_args()

    cfg = load_config()
    if cfg["data"]["dataset"] != "amazon":
        raise SystemExit("Only Amazon datasets are supported for now.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items, _, _ = dd.load_processed(cfg)
    raw = Path(cfg["paths"]["data_dir"]) / "raw"
    item_ids = items.sort_values("item_idx")["item_id"].to_numpy()

    texts = amazon_texts(raw, cfg["data"], args.meta, item_ids)
    missing = sum(t == "" for t in texts)
    print(f"{len(texts):,} items, {missing:,} without text ({missing / len(texts):.1%})")

    emb = embed(texts, device)
    out = Path(cfg["paths"]["data_dir"]) / "processed"
    pd.DataFrame({"item_idx": np.arange(len(texts)), "text": texts}).to_parquet(out / "item_text.parquet")
    np.save(out / "item_emb.npy", emb)
    print(f"Saved embeddings {emb.shape} to {out}")

    # Sanity check: nearest neighbours by text for a few items
    rng = np.random.default_rng(0)
    for i in rng.choice(len(texts), 3, replace=False):
        near = np.argsort(-(emb @ emb[i]))[1:4]
        print(f"\n{texts[i][:70]}")
        for j in near:
            print(f"   ~ {texts[j][:70]}")


if __name__ == "__main__":
    main()