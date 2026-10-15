"""Download MovieLens-1M into <data_dir>/raw.

Source: https://grouplens.org/datasets/movielens/1m/

    TRAJRISK_CONFIG=configs/movielens.yaml python scripts/download_movielens.py
"""

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from trajrisk.config import load_config

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["data_dir"]) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "ml-1m.zip"

    if target.exists():
        print("ml-1m.zip: already downloaded, skipping")
    else:
        print("ml-1m.zip: downloading...")
        partial = raw_dir / "ml-1m.zip.part"
        urllib.request.urlretrieve(URL, partial)
        partial.rename(target)

    info = {"url": URL, "size_mb": round(target.stat().st_size / 1024**2, 2),
            "sha256": sha256(target),
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(raw_dir / "download_manifest.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
