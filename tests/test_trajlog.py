"""Proposal section 4: trajectory logs (trajlog.py)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trajrisk import trajlog as tl  # noqa: E402
from trajrisk.evaluate import target_ranks  # noqa: E402


class _Diffusion:
    steps = 5
    alphas_cumprod = np.array([0.99, 0.95, 0.9, 0.8, 0.7])


def test_noise_schedule_goes_from_noisiest_to_final():
    rows = tl.noise_schedule(_Diffusion())
    assert [r["t"] for r in rows] == [4, 3, 2, 1, 0]
    assert rows[0]["normalized_t"] == 1.0 and rows[-1]["normalized_t"] == 0.0
    assert rows[0]["noise_std"] > rows[-1]["noise_std"]


def test_log_batch_top_items_ranks_and_summaries():
    torch.manual_seed(0)
    traj = torch.randn(5, 3, 20)
    seen = torch.zeros(3, 20, dtype=torch.bool)
    seen[:, 0] = True
    targets = torch.tensor([1, 2, 3])
    logged, ranks = tl.log_batch(traj, seen, targets, top_m=8)
    assert logged["items"].shape == (3, 5, 8)
    assert not (logged["items"] == 0).any()                       # seen item never listed
    expect = traj.masked_fill(seen.unsqueeze(0), float("-inf")).topk(8, dim=-1)
    assert np.array_equal(logged["items"][1, 2], expect.indices[2, 1].numpy())
    assert np.allclose(logged["dist_to_final"][:, -1], 0)         # last checkpoint = final
    assert np.allclose(logged["final_norm"], traj[-1].norm(dim=-1).numpy())
    assert np.array_equal(ranks[:, -1], target_ranks(traj[-1], targets, seen).numpy())
    assert "ranks" not in logged                                  # labels kept apart
