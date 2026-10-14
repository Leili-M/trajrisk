"""Stage 2: build and save the processed dataset.

    python scripts/prepare_data.py
"""

import json

from trajrisk.config import load_config
from trajrisk.prepare import build_dataset, save_dataset


def main():
    cfg = load_config()
    items, inter, cases, report = build_dataset(cfg)
    out = save_dataset(cfg, items, inter, cases, report)
    print(json.dumps(report["attrition"], indent=2))
    print(f"Recipes: {report['recipes_kept']:,}  "
          f"Train users: {report['train_users']:,}  "
          f"Train positives: {report['train_positives']:,}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
