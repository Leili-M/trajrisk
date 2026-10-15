"""Load the project settings.

By default configs/base.yaml is read. Set the environment variable
TRAJRISK_CONFIG to use another settings file (e.g. configs/movielens.yaml).
"""

import os
from pathlib import Path

import yaml

# The project folder: two levels above this file (src/trajrisk/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    """Read the settings file and return it as a dictionary."""
    if path is None:
        path = os.environ.get("TRAJRISK_CONFIG", "configs/base.yaml")
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Environment variables can replace the paths (useful on a GPU cluster)
    if os.environ.get("TRAJRISK_DATA_DIR"):
        cfg["paths"]["data_dir"] = os.environ["TRAJRISK_DATA_DIR"]
    if os.environ.get("TRAJRISK_OUTPUT_DIR"):
        cfg["paths"]["output_dir"] = os.environ["TRAJRISK_OUTPUT_DIR"]

    # Turn relative paths into full paths that start at the project folder
    for key in ("data_dir", "output_dir"):
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        cfg["paths"][key] = p

    cfg["config_file"] = str(path)
    return cfg
