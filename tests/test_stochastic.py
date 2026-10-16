"""The new sampler must agree with the official DiffRec code (needs torch and external/DiffRec)."""

import pytest

torch = pytest.importorskip("torch")

from trajrisk.diffrec_model import DIFFREC_DIR, build_diffrec  # noqa: E402
from trajrisk.stochastic import case_signals, sample_trajectory  # noqa: E402

pytestmark = pytest.mark.skipif(not DIFFREC_DIR.exists(), reason="external/DiffRec missing")

CFG = {"model": {"dims": [16], "emb_size": 4, "steps": 5, "noise_schedule": "linear-var",
                 "noise_scale": 1.0, "noise_min": 0.001, "noise_max": 0.2}}


def _setup(n_items=80, batch=8):
    torch.manual_seed(0)
    model, diffusion = build_diffrec(CFG, n_items, "cpu")
    model.eval()
    history = (torch.rand(batch, n_items) < 0.3).float()
    return model, diffusion, history


def test_noise_free_path_equals_official_recommendation():
    model, diffusion, h = _setup()
    ours = sample_trajectory(model, diffusion, h, noisy_start=False, sampling_noise=False)[-1]
    official = diffusion.p_sample(model, h, 0, False)
    assert torch.allclose(ours, official, atol=1e-6)


def test_noisy_path_equals_official_sampling_with_same_seed():
    model, diffusion, h = _setup()
    torch.manual_seed(123)
    ours = sample_trajectory(model, diffusion, h, noisy_start=True, sampling_noise=True)[-1]
    torch.manual_seed(123)
    official = diffusion.p_sample(model, h, diffusion.steps, True)
    assert torch.allclose(ours, official, atol=1e-6)


def test_different_seeds_give_different_answers():
    model, diffusion, h = _setup()
    torch.manual_seed(1)
    a = sample_trajectory(model, diffusion, h)[-1]
    torch.manual_seed(2)
    b = sample_trajectory(model, diffusion, h)[-1]
    assert not torch.allclose(a, b)


def test_one_noise_free_sample_has_zero_disagreement():
    model, diffusion, h = _setup()
    seen = h > 0
    targets = torch.zeros(h.shape[0], dtype=torch.long)
    sig = case_signals(model, diffusion, h, seen, targets, n_samples=1,
                       noisy_start=False, sampling_noise=False)
    assert sig["sample_disagree@10"].abs().max() == 0
    assert torch.equal(sig["miss"], sig["miss_one_sample"])
