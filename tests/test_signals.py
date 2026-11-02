import inspect

import numpy as np
import pytest

from trajrisk import signals as sg

K = 10


def lists(*rows):
    return np.array(rows, dtype=np.int32)


def test_identical_lists_give_zero():
    a = lists(list(range(K)))
    assert sg.set_change(a, a)[0] == 0
    assert sg.rank_change(a, a)[0] == 0


def test_disjoint_lists():
    a, b = lists(list(range(K))), lists(list(range(100, 100 + K)))
    assert sg.set_change(a, b)[0] == 1
    # every item displaced to rank K+1 on the other side: 2 * K(K+1)/2 over K * 2K
    assert sg.rank_change(a, b)[0] == pytest.approx((K + 1) / (2 * K))


def test_swap_changes_rank_not_set():
    a = lists(list(range(K)))
    b = a.copy()
    b[0, [0, 1]] = b[0, [1, 0]]
    assert sg.set_change(a, b)[0] == 0
    assert sg.rank_change(a, b)[0] == pytest.approx(2 / (K * K))


def test_one_replacement_matches_hand_count():
    a = lists(list(range(K)))
    b = a.copy()
    b[0, -1] = 99                       # last item replaced
    assert sg.set_change(a, b)[0] == pytest.approx(0.1)
    # item 9 at rank 10 -> K+1 (1), item 99 K+1 -> rank 10 (1); union 11
    assert sg.rank_change(a, b)[0] == pytest.approx(2 / (K * (K + 1)))


def test_weights():
    for rho in sg.RHOS:
        w = sg.late_weights(4, rho)
        assert w.sum() == pytest.approx(1)
        assert np.all(np.diff(w) >= -1e-12)          # never less weight on later steps
    assert np.allclose(sg.late_weights(4, 1.0), 0.25)


def make_log(n=6, n_ck=5, m=50, seed=0):
    rng = np.random.default_rng(seed)
    items = np.stack([np.stack([rng.permutation(200)[:m] for _ in range(n_ck)])
                      for _ in range(n)]).astype(np.int32)
    scores = -np.sort(-rng.random((n, n_ck, m)), axis=2).astype(np.float32)
    dist = rng.random((n, n_ck))
    dist[:, -1] = 0
    return items, scores, dist, rng.random(n) + 1


def test_compute_signals_shapes_and_direction():
    items, scores, dist, norm = make_log()
    items[0] = items[0, -1]                          # case 0: every checkpoint = final
    sig = sg.compute_signals(items, scores, dist, norm, score_scale=0.1)
    for group in sg.SIGNAL_GROUPS.values():
        for name in group:
            assert sig[name].shape == (6,)
            assert np.isfinite(sig[name]).all()
    assert sig["u_set_unif"][0] == 0 and sig["u_rank_unif"][0] == 0
    assert sig["cons_set"][0] == 0 and sig["reversal_set"][0] == 0
    assert (sig["u_set_unif"][1:] > 0).all()
    assert (sig["small_margin"] <= 0).all()         # negated margins
    e = sig["entropy_t1"]
    assert ((e >= 0) & (e <= 1 + 1e-9)).all()


def test_giffcf_length_trajectory_works():
    items, scores, dist, norm = make_log(n_ck=4)     # GiffCF: t=3,2,1 + final
    sig = sg.compute_signals(items, scores, dist, norm, score_scale=0.1)
    assert sig["u_set_early"].shape == (6,)


def test_reversal_detects_return():
    a, b = list(range(K)), list(range(100, 100 + K))
    items = np.zeros((1, 3, K + 1), dtype=np.int32)  # M = K + 1 (margin needs item K+1)
    items[0, :, :K] = [a, b, a]                      # leaves and comes back
    items[0, :, K] = 500
    scores = np.tile(np.linspace(1, 0, K + 1, dtype=np.float32), (1, 3, 1))
    sig = sg.compute_signals(items, scores, np.zeros((1, 3)), np.ones(1), 0.1, k=K)
    assert sig["u_set_early"][0] == 0                 # first checkpoint equals final
    assert sig["reversal_set"][0] == pytest.approx(2.0)


def test_signal_code_never_receives_labels():
    params = inspect.signature(sg.compute_signals).parameters
    assert not {"target", "targets", "labels", "target_rank", "miss"} & set(params)
