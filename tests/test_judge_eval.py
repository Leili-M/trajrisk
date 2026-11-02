import numpy as np
import pandas as pd
import pytest

from trajrisk import judge_eval as je


def test_labels_from_ranks():
    miss, loss = je.labels_from_ranks(np.array([0, 1, 9, 10, 500]))
    assert miss.tolist() == [False, False, False, True, True]
    assert loss[0] == pytest.approx(0.0)                 # rank 1: NDCG = 1
    assert loss[1] == pytest.approx(1 - 1 / np.log2(3))
    assert loss[3] == 1.0 and loss[4] == 1.0


def test_risk_coverage_oracle_random_order():
    loss = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=float)
    ref = je.reference_aurc(loss)
    assert ref["oracle"] < ref["random"] + 1e-9
    assert je.aurc(-loss, loss) > je.aurc(loss, loss)    # worst order > oracle
    cov, risk = je.risk_coverage(loss, loss)
    assert cov[-1] == 1.0 and risk[-1] == pytest.approx(loss.mean())


def test_judge_uses_signal_and_rejects_labels():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    miss = rng.random(n) < 1 / (1 + np.exp(-2 * x))
    df = pd.DataFrame({"sig": x, "noise": rng.normal(size=n), "miss": miss,
                       "user_id": np.arange(n) % 300})
    prob, _ = je.fit_predict(df.iloc[:1000], df.iloc[1000:], ["sig"])
    assert je.scores(prob, df["miss"].iloc[1000:], df["miss"].iloc[1000:].astype(float))["auroc"] > 0.7
    with pytest.raises(ValueError):
        je.fit_predict(df, df, ["sig", "miss"])
    oof = je.cv_predict(df, ["sig"], df["user_id"])
    assert oof.shape == (n,)


def test_reliability_and_interval():
    p = np.linspace(0.01, 0.99, 100)
    t = je.reliability_table(p, p > 0.5, n_bins=5)
    assert t["cases"].sum() == 100 and len(t) == 5
    m, lo, hi = je.t_interval([0.03, 0.04, 0.05])
    assert lo < m < hi and m == pytest.approx(0.04)
