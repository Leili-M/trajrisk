"""Proposal section 4: sampling DiffRec with fewer steps (respaced.py)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trajrisk import respaced as rs  # noqa: E402


def test_step_subsets_for_five_steps():
    s = rs.step_subsets(5)
    assert s[5] == [4, 3, 2, 1, 0]
    assert s[3] == [4, 2, 0]
    assert s[2] == [4, 0]
    assert all(v[0] == 4 and v[-1] == 0 for v in s.values())


def test_normalized_level():
    assert rs.normalized_level(4, 5) == 1.0 and rs.normalized_level(0, 5) == 0.0
    assert rs.normalized_level(2, 5) == 0.5


def test_rejects_bad_step_lists():
    with pytest.raises(ValueError):
        rs.respaced_trajectory(None, None, torch.zeros(1, 3), [0, 4])


class _Diff:
    """Minimal diffusion: linear betas, as numpy (like the official code)."""
    steps = 5
    betas = np.linspace(1e-4, 5e-3, 5)
    alphas_cumprod = np.cumprod(1 - betas)


def test_full_respacing_equals_standard_posterior_chain():
    torch.manual_seed(0)
    w = torch.randn(6, 6)
    model = lambda x, t: torch.tanh(x @ w + t[:, None].float())   # depends on x and t
    h = torch.rand(3, 6)
    d = _Diff()
    got = rs.respaced_trajectory(model, d, h, [4, 3, 2, 1, 0])
    # standard DDPM posterior with the original betas
    ac, b = d.alphas_cumprod, d.betas
    acp = np.append(1.0, ac[:-1])
    x, want = h, []
    for t in [4, 3, 2, 1, 0]:
        x0 = model(x, torch.full((3,), t))
        want.append(x0)
        x = b[t] * np.sqrt(acp[t]) / (1 - ac[t]) * x0 + (1 - acp[t]) * np.sqrt(1 - b[t]) / (1 - ac[t]) * x
    assert torch.allclose(got, torch.stack(want), atol=1e-5)


def test_fewer_steps_still_end_at_a_clean_guess():
    model = lambda x, t: x * 0.5 + 1.0
    out = rs.respaced_trajectory(model, _Diff(), torch.ones(2, 4), [4, 0])
    assert out.shape == (2, 2, 4)


# On Roihu, with the official DiffRec code: all steps reproduce the official trajectory
try:
    from trajrisk.diffrec_model import DIFFREC_DIR, build_diffrec
    from trajrisk.trajectory import diffrec_trajectory
    HAVE_DIFFREC = DIFFREC_DIR.exists()
except Exception:        # pragma: no cover
    HAVE_DIFFREC = False


@pytest.mark.skipif(not HAVE_DIFFREC, reason="external/DiffRec missing")
def test_full_respacing_reproduces_official_diffrec_trajectory():
    cfg = {"model": {"dims": [16], "emb_size": 4, "steps": 5, "noise_schedule": "linear-var",
                     "noise_scale": 0.0001, "noise_min": 0.0005, "noise_max": 0.005}}
    torch.manual_seed(0)
    model, diffusion = build_diffrec(cfg, 40, "cpu")
    model.eval()
    h = (torch.rand(6, 40) < 0.3).float()
    official = diffrec_trajectory(model, diffusion, h)
    ours = rs.respaced_trajectory(model, diffusion, h, rs.step_subsets(5)[5])
    assert torch.allclose(official, ours, atol=1e-5)
