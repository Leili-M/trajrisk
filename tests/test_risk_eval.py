import numpy as np
import pandas as pd

from trajrisk import risk_eval


def _cases(n=4000, seed=0):
    """Synthetic cases: longer history -> more often wrong (like MovieLens)."""
    rng = np.random.default_rng(seed)
    history = rng.integers(1, 500, n)
    hidden = rng.normal(size=n)                       # real per-case difficulty
    p_wrong = 1 / (1 + np.exp(-(np.log(history) - 3 + 1.5 * hidden)))
    df = pd.DataFrame({
        "history_size": history,
        "miss": rng.random(n) < p_wrong,
        # only a disguised copy of history length
        "proxy": -np.log(history) + rng.normal(scale=0.05, size=n),
        # carries the hidden difficulty
        "informative": hidden + rng.normal(scale=0.5, size=n),
        "constant": np.zeros(n),
    })
    users = rng.integers(0, 400, n)
    return df, users


def test_proxy_signal_has_no_value_beyond_history():
    df, users = _cases()
    res = risk_eval.evaluate(df, {"proxy": ["proxy"]}, users)
    single = res["single"].set_index("signal")
    assert single.loc["proxy", "auroc"] < 0.45              # looks useful, reversed
    assert single.loc["proxy", "spearman_with_history"] < -0.95
    # Inside groups it moves towards chance, but 4 coarse groups do not remove
    # history completely; the cross-validated added value below is the decisive check.
    within = res["within_groups"]["proxy"]
    assert (abs(within - 0.5) < abs(single.loc["proxy", "auroc"] - 0.5)).all()
    gain = res["added_value"].set_index("predictors").loc["history + proxy", "gain"]
    assert abs(gain) < 0.01


def test_informative_signal_adds_value():
    df, users = _cases()
    res = risk_eval.evaluate(df, {"informative": ["informative"]}, users)
    assert (res["within_groups"]["informative"] > 0.6).all()
    gain = res["added_value"].set_index("predictors").loc["history + informative", "gain"]
    assert gain > 0.05


def test_constant_signals_are_skipped():
    df, users = _cases(n=1000)
    res = risk_eval.evaluate(df, {"c": ["constant"], "p": ["proxy"]}, users)
    assert "constant" not in res["single"]["signal"].tolist()
    assert "history + c" not in res["added_value"]["predictors"].tolist()
