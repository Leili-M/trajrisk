"""Build the official DiffRec model, used unchanged from external/DiffRec."""

import sys

from .config import PROJECT_ROOT

DIFFREC_DIR = PROJECT_ROOT / "external" / "DiffRec" / "DiffRec"


def _import_diffrec():
    if not DIFFREC_DIR.exists():
        raise FileNotFoundError(
            f"DiffRec code not found at {DIFFREC_DIR}. "
            "Clone https://github.com/YiyanXu/DiffRec into external/DiffRec "
            "and check out commit e42532f.")
    if str(DIFFREC_DIR) not in sys.path:
        sys.path.insert(0, str(DIFFREC_DIR))
    from models.DNN import DNN
    import models.gaussian_diffusion as gd
    return DNN, gd


def build_diffrec(cfg, n_items, device):
    """Return (model, diffusion) built from the settings in cfg['model']."""
    DNN, gd = _import_diffrec()
    m = cfg["model"]
    diffusion = gd.GaussianDiffusion(
        mean_type=gd.ModelMeanType.START_X,
        noise_schedule=m["noise_schedule"],
        noise_scale=m["noise_scale"], noise_min=m["noise_min"], noise_max=m["noise_max"],
        steps=m["steps"], device=device,
    ).to(device)
    out_dims = list(m["dims"]) + [n_items]
    model = DNN(out_dims[::-1], out_dims, emb_size=m["emb_size"],
                time_type="cat", norm=False).to(device)
    return model, diffusion


def recommend_scores(model, diffusion, history):
    """Final DiffRec scores for a dense batch of histories (no added noise, no sampling noise)."""
    return diffusion.p_sample(model, history, 0, False)
