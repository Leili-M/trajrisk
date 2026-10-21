"""Download one Amazon Reviews 2023 5-core 'rating only' file (McAuley Lab, UCSD).

    TRAJRISK_CONFIG=configs/amazon_video_games.yaml PYTHONPATH=src python scripts/download_amazon.py

Saves <data_dir>/raw/<raw_file> and a manifest with its sha256.
Source: https://amazon-reviews-2023.github.io/ (cite Hou et al., 2024, arXiv:2403.03952)
"""

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from trajrisk.config import load_config

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/"


def main():
    cfg = load_config()
    name = cfg["data"]["raw_file"]
    raw = Path(cfg["paths"]["data_dir"]) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    dest = raw / name

    if not dest.exists():
        part = dest.with_suffix(dest.suffix + ".part")
        print("Downloading", BASE + name)
        urllib.request.urlretrieve(BASE + name, part)
        part.rename(dest)
    else:
        print("Already downloaded:", dest)

    manifest = {
        "source": BASE + name,
        "file": name,
        "bytes": dest.stat().st_size,
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(raw / "download_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()