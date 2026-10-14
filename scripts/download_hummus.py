"""Download the preprocessed HUMMUS files into data/raw.

Source: https://gitlab.com/felix134/connected-recipe-data-set
Folder: data/hummus_data/preprocessed (last commit 5c410a5c, 2023-07-20)
"""

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from trajrisk.config import load_config

BASE_URL = (
    "https://gitlab.com/felix134/connected-recipe-data-set/-/raw/master/"
    "data/hummus_data/preprocessed/"
)
FILES = ["pp_reviews.zip", "pp_recipes.zip", "pp_members.zip"]


def sha256(path):
    """Fingerprint of a file: changes if even one byte changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["data_dir"]) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    files_info = {}
    for name in FILES:
        target = raw_dir / name
        if target.exists():
            print(f"{name}: already downloaded, skipping")
        else:
            print(f"{name}: downloading...")
            partial = raw_dir / (name + ".part")
            urllib.request.urlretrieve(BASE_URL + name, partial)
            partial.rename(target)

        size_mb = target.stat().st_size / 1024**2
        files_info[name] = {
            "url": BASE_URL + name,
            "size_mb": round(size_mb, 2),
            "sha256": sha256(target),
        }
        print(f"  size: {size_mb:.2f} MB")

    manifest = {
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files_info,
    }
    manifest_path = raw_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
