"""Proposal section 1: the risk score uses no future information and never the target.

R1.1  the history of a case contains only interactions strictly before the case time
R1.3  the last step of the trajectory is exactly the final DiffRec recommendation
R1.4  top-K lists never contain items the user has already seen
R1.7  changing the target changes no signal, only the miss label
R1.8  outcome columns can never be used as predictors
"""

import numpy as np
import pandas as pd
import pytest

from trajrisk import risk_eval
from trajrisk.diffrec_data import case_matrices


# ---------------------------------------------------------------- R1.1
def test_history_strictly_before_case_time_with_exact_timestamps():
    """Amazon and MovieLens have exact times: an event at the case time or later is excluded."""
    inter = pd.DataFrame({
        "user_id": [1, 1, 1, 1],
        "item_idx": [0, 1, 2, 3],
        "date": pd.to_datetime(["2019-01-01 10:00:00.000", "2019-01-01 11:59:59.999",
                                "2019-01-01 12:00:00.000", "2019-01-01 12:00:00.001"]),
        "positive": [True, True, True, True],
    })
    cases = pd.DataFrame({"user_id": [1], "date": pd.to_datetime(["2019-01-01 12:00:00.000"])})
    history, seen = case_matrices(cases, inter, n_items=4)
    assert history.toarray().tolist() == [[1, 1, 0, 0]]
    assert seen.toarray().tolist() == [[1, 1, 0, 0]]


def test_other_users_never_enter_a_history():
    inter = pd.DataFrame({
        "user_id": [1, 2], "item_idx": [0, 1],
        "date": pd.to_datetime(["2019-01-01", "2019-01-01"]), "positive": [True, True],
    })
    cases = pd.DataFrame({"user_id": [1], "date": pd.to_datetime(["2019-02-01"])})
    history, _ = case_matrices(cases, inter, n_items=2)
    assert history.toarray().tolist() == [[1, 0]]


# ---------------------------------------------------------------- R1.8
def _toy_frame(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"miss": rng.random(n) < 0.5, "signal": rng.random(n),
                         "history_size": rng.integers(1, 20, n)})


def test_outcome_columns_cannot_be_predictors():
    df = _toy_frame()
    users = np.arange(len(df))
    with pytest.raises(ValueError):
        risk_eval.cv_auroc(df, ["signal", "miss"], users)
    with pytest.raises(ValueError):
        risk_eval.cv_predict(df, ["miss"], users)
    with pytest.raises(ValueError):
        risk_eval.evaluate(df, {"bad": ["signal", "miss"]}, users)


def test_ordinary_signals_are_accepted():
    df = _toy_frame()
    p = risk_eval.cv_predict(df, ["signal"], np.arange(len(df)))
    assert p.shape == (len(df),)


# ---------------------------------------------------------------- R1.3, R1.4, R1.7 (need torch and DiffRec)
torch = pytest.importorskip("torch")

from trajrisk.diffrec_model import DIFFREC_DIR, build_diffrec, recommend_scores  # noqa: E402
from trajrisk.trajectory import diffrec_trajectory, topk_lists  # noqa: E402

needs_diffrec = pytest.mark.skipif(not DIFFREC_DIR.exists(), reason="external/DiffRec missing")


def test_topk_never_contains_seen_items():
    scores = torch.arange(10.0).repeat(3, 1)          # item 9 has the highest score
    seen = torch.zeros(3, 10, dtype=torch.bool)
    seen[:, 9] = True
    seen[:, 8] = True
    top = topk_lists(scores, seen, 3)
    assert not seen.gather(1, top).any()
    assert top[0].tolist() == [7, 6, 5]


def _setup(noise_scale, n_items=60, batch=8):
    cfg = {"model": {"dims": [16], "emb_size": 4, "steps": 5, "noise_schedule": "linear-var",
                     "noise_scale": noise_scale, "noise_min": 0.0005, "noise_max": 0.005}}
    torch.manual_seed(0)
    model, diffusion = build_diffrec(cfg, n_items, "cpu")
    model.eval()
    history = (torch.rand(batch, n_items) < 0.3).float()
    return model, diffusion, history


@needs_diffrec
@pytest.mark.parametrize("noise_scale", [0.0001, 1.0])   # 0.0001 is the setting of all final models
def test_last_trajectory_step_equals_final_recommendation(noise_scale):
    model, diffusion, h = _setup(noise_scale)
    traj = diffrec_trajectory(model, diffusion, h)
    assert traj.shape[0] == diffusion.steps
    assert torch.allclose(traj[-1], recommend_scores(model, diffusion, h), atol=1e-6)


@needs_diffrec
def test_target_changes_only_the_label_in_trajectory_signals():
    from trajrisk.stochastic import case_signals
    model, diffusion, h = _setup(0.0001)
    seen = h > 0
    t_a = torch.zeros(h.shape[0], dtype=torch.long)
    t_b = torch.randint(0, h.shape[1], (h.shape[0],))
    torch.manual_seed(0)
    a = case_signals(model, diffusion, h, seen, t_a, 1, noisy_start=False, sampling_noise=False)
    torch.manual_seed(0)
    b = case_signals(model, diffusion, h, seen, t_b, 1, noisy_start=False, sampling_noise=False)
    for k in a:
        if k not in risk_eval.LABEL_COLUMNS:
            assert torch.equal(a[k], b[k]), k


@needs_diffrec
def test_target_changes_only_the_label_in_masking_signals():
    from trajrisk.masking import masked_signals
    model, diffusion, h = _setup(0.0001)
    seen = h > 0
    t_a = torch.zeros(h.shape[0], dtype=torch.long)
    t_b = torch.randint(0, h.shape[1], (h.shape[0],))
    torch.manual_seed(0)
    a = masked_signals(model, diffusion, h, seen, t_a, 3, 0.5)
    torch.manual_seed(0)
    b = masked_signals(model, diffusion, h, seen, t_b, 3, 0.5)
    for k in a:
        if k not in risk_eval.LABEL_COLUMNS:
            assert torch.equal(a[k], b[k]), k
