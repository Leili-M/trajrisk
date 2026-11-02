import numpy as np
import pytest
import torch

from trajrisk import perturb as pt
from trajrisk import signals as sg


def test_torch_measures_equal_numpy_versions():
    rng = np.random.default_rng(0)
    a = np.stack([rng.permutation(30)[:10] for _ in range(50)])
    b = np.stack([rng.permutation(30)[:10] for _ in range(50)])
    b[:10] = a[:10]                                   # some identical lists
    b[10:20] = a[10:20][:, ::-1]                      # some reversed lists
    ta, tb = torch.as_tensor(a), torch.as_tensor(b)
    assert np.allclose(pt.set_change_t(ta, tb).numpy(), sg.set_change(a, b))
    assert np.allclose(pt.rank_change_t(ta, tb).numpy(), sg.rank_change(a, b), atol=1e-6)


class FakeDiffusion:
    """Deterministic without noise; with noise, scores get random perturbations."""
    steps = 5
    noise_scale = 1e-4

    def p_sample(self, model, x, steps, sampling_noise):
        base = x @ model
        if steps > 0 or sampling_noise:
            return base + 0.5 * torch.randn_like(base)
        return base


def test_perturbation_signals_shapes_and_reference_list():
    torch.manual_seed(0)
    n_users, n_items = 8, 40
    model = torch.rand(n_items, n_items)
    hist = (torch.rand(n_users, n_items) < 0.2).float()
    hist[:, 0] = 1                                    # never empty
    seen = hist > 0
    sig, top = pt.perturbation_signals(model, FakeDiffusion(), hist, seen, n_samples=10,
                                       n_masks=4, rate=0.5)
    ref_top = (hist @ model).masked_fill(seen, float("-inf")).topk(10, dim=-1).indices
    assert torch.equal(top, ref_top)                  # judged list = ordinary list
    for name in ["samp2_set", "samp5_rank", "samp10_score", "mask_set", "mask_rank",
                 "mask_score", "mask_unstable"]:
        assert sig[name].shape == (n_users,)
        assert torch.isfinite(sig[name]).all()
    assert (sig["samp10_set"] > 0).any()              # noise changes some lists
    assert ((sig["mask_set"] >= 0) & (sig["mask_set"] <= 1)).all()


def test_signature_has_no_labels():
    import inspect
    params = inspect.signature(pt.perturbation_signals).parameters
    assert not {"targets", "target", "labels", "miss"} & set(params)
