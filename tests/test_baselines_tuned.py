"""Proposal section 4: tuned ItemKNN and EASE baselines (baselines.py)."""

import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from trajrisk import baselines as bl  # noqa: E402


def _train(seed=0, users=40, items=12):
    rng = np.random.default_rng(seed)
    return sp.csr_matrix((rng.random((users, items)) < 0.3).astype(np.float32))


def test_gram_matrix_equals_xtx_for_any_block_size():
    X = _train()
    full = (X.T @ X).toarray()
    assert np.allclose(bl.gram_matrix(X, block=5), full)


def test_itemknn_keeps_k_neighbours_and_no_self_similarity():
    X = _train()
    G, c = bl.gram_matrix(X), bl.item_counts(X)
    h = torch.eye(12)                       # one history item per row -> row of W
    W = bl.item_knn_pruned(G, c, k=3, shrink=0, device="cpu")(h)
    assert (W.diagonal() == 0).all()
    assert ((W > 0).sum(dim=1) <= 3).all()
    # unpruned version equals plain cosine similarity
    full = bl.item_knn_pruned(G, c, k=None, shrink=0, device="cpu")(h).numpy()
    cos = G / np.sqrt(np.outer(c, c))
    np.fill_diagonal(cos, 0)
    assert np.allclose(full, cos, atol=1e-5)


def test_ease_matches_closed_form():
    X = _train()
    G = bl.gram_matrix(X)
    lam = 5.0
    P = np.linalg.inv(G + lam * np.eye(12))
    B = -P / np.diag(P)[None, :]
    np.fill_diagonal(B, 0)
    got = bl.ease(G, lam, device="cpu")(torch.eye(12)).numpy()
    assert np.allclose(got, B, atol=1e-4)
    assert np.allclose(np.diag(got), 0)
