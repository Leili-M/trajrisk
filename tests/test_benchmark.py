"""Proposal section 3: constraint rules, utility and scenario construction (benchmark.py)."""

import numpy as np
import pandas as pd

from trajrisk import benchmark as bm


def _attrs():
    """Four recipes covering satisfied, violated and unknown cases."""
    return pd.DataFrame({
        "item_idx": [0, 1, 2, 3],
        "tag:vegetarian": [True, True, False, None],
        "tag:vegan": [False, False, False, None],
        "tag:main-dish": [True, False, True, None],
        "mentions_meat": [False, True, False, False],
        "mentions_animal_product": [True, True, True, False],
        "duration": [20.0, 10.0, 45.0, np.nan],
        "n_ingredients": [5, 3, 9, 4], "calories": [300.0, 100.0, 600.0, np.nan],
        "protein": [10.0, 30.0, 5.0, 1.0], "fsa_score": [5.0, 6.0, 1.0, np.nan],
    })


def test_vegetarian_rule_needs_tag_and_no_meat():
    s = bm.constraint_status(_attrs(), "diet", "vegetarian")
    # tagged + no meat: ok; tagged but meat word: violated; no tag + no meat: unknown
    assert s[0] == 1 and s[1] == 0 and np.isnan(s[2]) and np.isnan(s[3])


def test_course_and_time_rules():
    a = _attrs()
    c = bm.constraint_status(a, "course", "main-dish")
    assert c[0] == 1 and c[1] == 0 and np.isnan(c[3])          # no tags at all: unknown
    t = bm.constraint_status(a, "max_time", 30)
    assert t[0] == 1 and t[2] == 0 and np.isnan(t[3])


def test_feasible_only_when_every_constraint_is_known_satisfied():
    hard = {"diet": "vegetarian", "max_time": 30}
    assert bm.feasible_mask(_attrs(), hard).tolist() == [True, False, False, False]
    assert bm.violations(_attrs(), hard).tolist() == [0, 1, 1, 0]


def test_utility_prefers_the_right_direction_and_ignores_unknown():
    a = _attrs()
    pref = bm.preference_scores(a)
    u_time = bm.item_utility(pref, {"time": 1.0})
    assert np.argmax(np.nan_to_num(u_time, nan=-1)) == 1        # shortest duration is best
    u = bm.item_utility(pref, {"calories": 0.5, "protein": 0.5})
    assert not np.isnan(u[3])                                    # calories unknown, protein known


def test_list_utility_regret_and_ideal_list():
    util = np.array([0.9, 0.8, 0.1, 0.5])
    feasible = np.array([True, False, True, True])
    best = bm.ideal_list(util, feasible, k=2)
    assert best.tolist() == [0, 3]
    assert bm.regret(best, util, feasible, k=2) == 0
    assert bm.regret([1, 2], util, feasible, k=2) > 0            # infeasible item gives 0 utility


def _catalog(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    a = pd.DataFrame({"item_idx": range(n)})
    for t in bm.DIETS + bm.COURSES + bm.CUISINES:
        a[f"tag:{t}"] = list(rng.random(n) < 0.15)
    a["mentions_meat"] = list(rng.random(n) < 0.3)
    a["mentions_animal_product"] = list(rng.random(n) < 0.5)
    a["duration"] = rng.choice([10, 20, 40, 90], n).astype(float)
    for c in ["n_ingredients", "calories", "protein", "fsa_score"]:
        a[c] = rng.random(n)
    return a


def test_scenarios_respect_holdout_types_and_users():
    a = _catalog()
    combos = bm.hard_combinations(a)
    cases = pd.DataFrame({"case_id": range(500), "user_id": np.arange(500) % 250,
                          "date": pd.Timestamp("2015-01-01")})
    rng = np.random.default_rng(0)
    val = bm.sample_scenarios(combos, cases, 40, "valid", rng)
    test = bm.sample_scenarios(combos, cases, 60, "test", rng,
                               exclude_users={s["user_id"] for s in val})
    assert not {s["user_id"] for s in val} & {s["user_id"] for s in test}   # unseen test users
    assert not any(s["heldout_soft"] or s["heldout_course"] for s in val)
    assert len({s["user_id"] for s in val}) == 40               # one scenario per user
    pref = bm.preference_scores(a)
    for s in val + test:
        full = bm.complete_scenario(s, a, pref)
        lo, hi = bm.FEASIBLE_TYPES[s["feasible_type"]]
        assert lo <= full["n_feasible"] <= hi
        assert full["abstain"] == (full["n_feasible"] == 0)
        assert set(full["ideal_list"]) <= set(full["feasible"])
