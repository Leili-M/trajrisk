"""Second backbone: PyTorch re-implementation of GiffCF (giffcf.py)."""

import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from trajrisk import giffcf as gc  # noqa: E402


def _train(users=30, items=15, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.random((users, items)) < 0.3).astype(np.float32)
    X[np.arange(users), rng.integers(0, items, users)] = 1      # no empty user
    X[rng.integers(0, users, items), np.arange(items)] = 1      # no empty item
    return sp.csr_matrix(X)


def test_graph_operator_matches_dense_formula():
    X = _train()
    R, lam = gc.graph_operator(X)
    D = X.toarray()
    Rd = np.diag(D.sum(1) ** -0.25) @ D @ np.diag(D.sum(0) ** -0.5)
    assert np.allclose(R.toarray(), Rd, atol=1e-5)
    assert np.isclose(lam, np.linalg.eigvalsh(Rd.T @ Rd).max(), rtol=1e-4)


def test_smooth_and_filter():
    X = _train()
    m = gc.build(X, {"embed_dim": 8}, "cpu")
    x = torch.from_numpy(X[:4].toarray())
    Rd = m.R.to_dense()
    want = x @ Rd.T @ Rd / m.lam
    assert torch.allclose(m.smooth(x), want, atol=1e-5)
    Ax = m.smooth(x)
    assert torch.allclose(m.filter(x, Ax, 0), x)                          # t = 0: no smoothing
    assert torch.allclose(m.filter(x, Ax, m.T), x + m.alpha * (Ax - x))  # t = T: full smoothing


def test_trajectory_shape_and_final_output():
    X = _train()
    m = gc.build(X, {"embed_dim": 8, "T": 3}, "cpu").eval()
    x = torch.from_numpy(X[:5].toarray())
    traj = m.trajectory(x)
    assert traj.shape == (4, 5, 15)                                       # t = 3, 2, 1 and final
    assert torch.allclose(traj[-1], m.recommend(x))
    assert len(m.checkpoints()) == 4


def test_training_reduces_the_loss():
    X = _train(users=60)
    m = gc.build(X, {"embed_dim": 8}, "cpu")
    x = torch.from_numpy(X.toarray())
    torch.manual_seed(0)
    before = m.training_loss(x).item()
    best, epoch, hist = gc.fit(m, X, lambda mod: -mod.training_loss(x).item(), epochs=30,
                               batch_size=20, val_every=5, patience=100, device="cpu")
    torch.manual_seed(0)
    assert m.training_loss(x).item() < before
