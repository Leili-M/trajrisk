import numpy as np
import pandas as pd
import pytest

from trajrisk import preference as pf
from trajrisk import preference_eval as pe


def toy(n=30, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.random((n, 5))
    P[3, 2] = np.nan                                   # one unknown attribute
    util = rng.random(n)
    viol = (rng.random(n) < 0.2).astype(int)
    feasible = viol == 0
    return P, util, viol, feasible


def test_rerank_soft_and_hard():
    P, util, viol, feasible = toy()
    z = np.linspace(1, 0, len(util))
    seen = np.zeros(len(util), bool)
    seen[0] = True
    soft = pf.rerank(z, util, viol, 0.0, 0.0, seen)
    assert soft[0] == 1 and 0 not in soft              # seen item never recommended
    strict = pf.rerank(z, util, viol, 0.0, 100.0, seen)
    assert (viol[strict] == 0).all()                   # large penalty removes violations
    hard = pf.rerank(z, util, viol, 0.0, 0.0, seen, feasible=feasible)
    assert feasible[hard[hard >= 0]].all()
    none = pf.rerank(z, util, viol, 0.0, 0.0, seen, feasible=np.zeros(len(util), bool))
    assert (none == -1).all()                          # nothing feasible: abstain


def test_distances_separate_identity_from_preference():
    P, util, viol, feasible = toy()
    w = np.array([0.6, 0.4, 0, 0, 0])
    final = np.arange(10)
    same_set = final[::-1].copy()
    assert pf.attr_distance(P, w, same_set, final) == pytest.approx(0)
    # replace one item by a recipe with identical relevant attributes
    P2 = P.copy()
    P2[20, :2] = P2[9, :2]
    swap = final.copy()
    swap[9] = 20
    assert pf.attr_distance(P2, w, swap, final) == pytest.approx(0, abs=1e-12)
    far = np.arange(10, 20)
    assert pf.attr_distance(P, w, far, final) > 0
    # unequal lengths: missing items cost 1
    short = np.concatenate([final[:5], np.full(5, -1)])
    assert pf.attr_distance(P, w, short, final) == pytest.approx(0.5)


def test_scenario_signals_and_outcome():
    P, util, viol, feasible = toy()
    w = np.array([1.0, 0, 0, 0, 0])
    stable = np.tile(np.arange(10), (5, 1))
    sig = pf.scenario_signals(stable, P, w, util, viol, ~feasible & (viol == 0))
    for k in ["u_set", "u_rank", "u_attr", "u_util", "u_attr_rank", "u_dist", "u_generic", "viol_traj"]:
        assert sig[k] == pytest.approx(0)
    moving = stable.copy()
    moving[:-1] = np.arange(10, 20)
    sig2 = pf.scenario_signals(moving, P, w, util, viol, np.zeros(len(util), bool))
    assert sig2["u_set"] == pytest.approx(1) and sig2["u_attr"] > 0
    out = pf.scenario_outcome(stable[-1], util, feasible, viol)
    assert 0 <= out["regret"] <= 1 and out["list_len"] == 10
    assert out["failure"] == (out["regret"] >= pf.FAIL_REGRET)


def fake_rows(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for run in ["a", "b"]:
        for i in range(120):
            fail = i % 3 == 0
            for pipe in ["plain", "soft", "hard"]:
                rows.append({"run": run, "scenario_id": f"s{i}", "split": "valid" if i < 60 else "test",
                             "pipeline": pipe, "feasible_type": "none" if i % 10 == 9 else "many",
                             "soft_type": "single", "heldout": i % 7 == 0,
                             "log_history": rng.random(), "diversity": rng.random(),
                             "u_set": rng.random() + 0.3 * fail, "u_rank": rng.random(),
                             "u_attr": rng.random() + 0.6 * fail, "u_util": rng.random() + 0.5 * fail,
                             "u_attr_rank": rng.random(), "u_dist": rng.random(),
                             "u_generic": rng.random(), "viol_traj": rng.random(),
                             "viol_final": rng.random(), "unknown_final": rng.random(),
                             "util_final": rng.random(), "regret": 0.7 if fail else 0.2,
                             "n_violating": int(rng.random() < 0.2), "list_len": 10,
                             "failure": fail})
    return pd.DataFrame(rows)


def test_evaluation_tables_run():
    df = fake_rows()
    assert len(pe.outcome_table(df)) == 3
    single, n_fail, n = pe.single_signal_table(df, "soft")
    attr = single.set_index("signal").loc["u_attr"]
    assert attr["auroc"] > 0.7 and 0 <= attr["false alarms among suitable"] <= 1
    judge, gains = pe.judge_table(df, "soft", b=50)
    assert len(judge) == 4 and "final checks + preference over final checks" in gains
    subsets, med = pe.subset_table(df, "soft")
    assert subsets["cases"].sum() == n


def test_percentile_scale_and_new_tables():
    scores = np.array([5.0, 1.0, 3.0, 100.0])
    seen = np.array([False, False, True, False])
    p = pf.percentile(scores, seen)
    assert p[3] == pytest.approx(1.0) and p[1] == pytest.approx(1 / 3) and p[2] == 0
    df = fake_rows()
    corr = pe.regret_correlation_table(df, "soft")
    assert corr.set_index("signal").loc["u_attr", "spearman with regret"] > 0
    flags = pe.flag_rate_table(df, "soft")
    row = flags.set_index("signal").loc["u_attr"]
    assert 0 < row["flag rate"] < 1 and row["regret flagged"] > row["regret not flagged"]
