
"""Read the settings file and resolve folder paths."""



import os

from pathlib import Path



import yaml



# The project folder: two levels above this file (src/trajrisk/config.py)

PROJECT_ROOT = Path(__file__).resolve().parents[2]





def storage_root():

    """Where data and outputs live.



    Default: the project folder (Colab). On a cluster, set TRAJRISK_STORAGE_ROOT

    to a large disk (for example /scratch/<project>); relative data_dir and

    output_dir in the settings file then start there.

    """

    root = os.environ.get("TRAJRISK_STORAGE_ROOT")

    return Path(root) if root else PROJECT_ROOT





def load_config(path=None):

    """Read the settings file and return it as a dictionary."""

    if path is None:

        path = os.environ.get("TRAJRISK_CONFIG", "configs/base.yaml")

    path = Path(path)

    if not path.is_absolute():

        path = PROJECT_ROOT / path

    with open(path) as f:

        cfg = yaml.safe_load(f)



    # Environment variables can replace the paths completely

    if os.environ.get("TRAJRISK_DATA_DIR"):

        cfg["paths"]["data_dir"] = os.environ["TRAJRISK_DATA_DIR"]

    if os.environ.get("TRAJRISK_OUTPUT_DIR"):

        cfg["paths"]["output_dir"] = os.environ["TRAJRISK_OUTPUT_DIR"]



    # Relative paths start at the storage root (the project folder by default)

    root = storage_root()

    for key in ("data_dir", "output_dir"):

        p = Path(cfg["paths"][key])

        if not p.is_absolute():

            p = root / p

        cfg["paths"][key] = p



    cfg["config_file"] = str(path)

    return cfg

