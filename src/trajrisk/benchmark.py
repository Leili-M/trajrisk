"""Controlled benchmark (proposal section 3): constraint rules, utility and scenarios.

Every rule used to judge a recommendation in the benchmark lives here, so the
scenario builder (section 3), the request-aware reranker (section 6) and the
evaluation (section 8) all use exactly the same definitions.

Hard constraints are checked by explicit rules on real recipe attributes. Each
constraint gives one of three answers per recipe:
    1.0  satisfied        0.0  violated        NaN  unknown (attribute missing)
A recipe is feasible only if every constraint is known to be satisfied.

Soft utility is synthetic: a weighted mean of preference scores, where each
preference score is the recipe's percentile in the catalog, oriented so that
1 is best. It is not a measure of real user satisfaction.
"""

import itertools

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- vocabulary
DIETS = ["vegetarian", "vegan"]
COURSES = ["main-dish", "desserts", "breakfast", "lunch", "appetizers",
           "salads", "soups-stews", "side-dishes"]
CUISINES = ["north-american", "european", "asian", "mexican", "italian", "indian"]
MAX_TIMES = [15, 30, 60]

# soft preference -> (attribute column, direction: -1 smaller is better, +1 larger is better)
PREFERENCES = {"time": ("duration", -1), "ingredients": ("n_ingredients", -1),
               "calories": ("calories", -1), "protein": ("protein", +1),
               "health": ("fsa_score", +1)}

# Preference patterns. Conflicting pairs have a clearly negative rank correlation
# in the catalog (reports/benchmark/attributes.md): calories vs protein -0.72,
# protein vs health -0.37, time vs protein -0.31. Aligned pairs are positive.
SOFT_PATTERNS = {
    "single": [("time",), ("ingredients",), ("calories",), ("protein",), ("health",)],
    "aligned": [("time", "ingredients"), ("calories", "health"), ("time", "calories")],
    "conflicting": [("calories", "protein"), ("protein", "health"), ("time", "protein")],
}
# Held out from validation, used only in test scenarios (proposal: hold out
# preference combinations and, where feasible, item categories).
HELDOUT_SOFT = [("time", "calories"), ("protein", "health")]
HELDOUT_COURSE = "soups-stews"

# Feasible-set sizes for the three scenario types required by the proposal
FEASIBLE_TYPES = {"none": (0, 0), "few": (10, 50), "many": (201, 10**9)}


# ---------------------------------------------------------------- constraint rules
def _bool(col):
    """Object column of True / False / None -> float array 1 / 0 / NaN."""
    return np.array([np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
                     for v in col], dtype=float)


def constraint_status(attrs, kind, value):
    """1 satisfied, 0 violated, NaN unknown, for every recipe in `attrs`.

    diet:    satisfied if tagged and no conflicting ingredient word; violated if the
             ingredients mention meat (vegetarian) or an animal product (vegan).
    course / cuisine: satisfied if tagged; violated if the recipe has tags but not
             this one; unknown if the recipe has no tags at all.
    max_time: from the recorded duration in minutes.
    """
    n = len(attrs)
    if kind == "diet":
        tag = _bool(attrs[f"tag:{value}"])
        bad = _bool(attrs["mentions_meat" if value == "vegetarian" else "mentions_animal_product"])
        out = np.full(n, np.nan)
        out[(tag == 1) & (bad == 0)] = 1.0
        out[bad == 1] = 0.0
        return out
    if kind in ("course", "cuisine"):
        tag = _bool(attrs[f"tag:{value}"])
        return tag                               # 1 tagged, 0 tagged-but-other, NaN no tags
    if kind == "max_time":
        d = attrs["duration"].to_numpy(dtype=float)
        out = np.where(np.isnan(d), np.nan, (d <= value).astype(float))
        return out
    raise ValueError(f"unknown constraint kind {kind}")


def hard_status(attrs, hard):
    """Matrix (recipes x constraints) of 1 / 0 / NaN for a dict of hard constraints."""
    cols = [constraint_status(attrs, k, v) for k, v in hard.items() if v is not None]
    return np.column_stack(cols) if cols else np.ones((len(attrs), 0))


def feasible_mask(attrs, hard):
    s = hard_status(attrs, hard)
    return (s == 1).all(axis=1)


def violations(attrs, hard):
    """Number of constraints each recipe is known to violate (c_q in proposal formula 9)."""
    return (hard_status(attrs, hard) == 0).sum(axis=1)


# ---------------------------------------------------------------- soft utility
def preference_scores(attrs):
    """Percentile of every recipe for each preference, oriented so that 1 = best."""
    out = {}
    for name, (col, sign) in PREFERENCES.items():
        r = attrs[col].rank(pct=True)
        out[name] = (r if sign > 0 else 1 - r + 1 / r.notna().sum()).to_numpy()
    return pd.DataFrame(out, index=attrs.index)


def item_utility(pref, soft):
    """Weighted mean of the preference scores; weights are renormalised over the
    attributes that are known for each recipe (a missing attribute is unknown, not bad)."""
    w = pd.Series(soft, dtype=float)
    vals = pref[w.index].to_numpy()
    known = ~np.isnan(vals)
    num = np.nansum(vals * w.to_numpy(), axis=1)
    den = (known * w.to_numpy()).sum(axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def list_utility(items, utility, feasible, k=10):
    """Prespecified list utility: discounted sum of item utilities over the top-k,
    where an infeasible (or unknown) recipe contributes 0."""
    items = list(items)[:k]
    gains = [utility[i] if feasible[i] and not np.isnan(utility[i]) else 0.0 for i in items]
    return float(sum(g / np.log2(r + 2) for r, g in enumerate(gains)))


def ideal_list(utility, feasible, k=10):
    """Best feasible list under the scenario utility (reference for regret)."""
    idx = np.flatnonzero(feasible & ~np.isnan(utility))
    return idx[np.argsort(-utility[idx], kind="stable")][:k]


def regret(items, utility, feasible, k=10):
    """1 - list utility / ideal list utility (0 = best possible feasible list)."""
    best = list_utility(ideal_list(utility, feasible, k), utility, feasible, k)
    return float("nan") if best == 0 else 1 - list_utility(items, utility, feasible, k) / best


# ---------------------------------------------------------------- scenario construction
def hard_combinations(attrs, max_constraints=4):
    """All requests built from 1-4 hard constraints, with their number of feasible recipes."""
    single = {}
    for d in DIETS:
        single[("diet", d)] = constraint_status(attrs, "diet", d) == 1
    for c in COURSES:
        single[("course", c)] = constraint_status(attrs, "course", c) == 1
    for c in CUISINES:
        single[("cuisine", c)] = constraint_status(attrs, "cuisine", c) == 1
    for t in MAX_TIMES:
        single[("max_time", t)] = constraint_status(attrs, "max_time", t) == 1
    rows = []
    for d, c, k, t in itertools.product([None] + DIETS, [None] + COURSES,
                                        [None] + CUISINES, [None] + MAX_TIMES):
        hard = {"diet": d, "course": c, "cuisine": k, "max_time": t}
        used = [(kind, v) for kind, v in hard.items() if v is not None]
        if not 1 <= len(used) <= max_constraints:
            continue
        m = np.ones(len(attrs), dtype=bool)
        for key in used:
            m &= single[key]
        rows.append({**hard, "n_constraints": len(used), "n_feasible": int(m.sum())})
    combos = pd.DataFrame(rows)
    combos["feasible_type"] = None
    for name, (lo, hi) in FEASIBLE_TYPES.items():
        combos.loc[combos["n_feasible"].between(lo, hi), "feasible_type"] = name
    # A request with no feasible recipe must still be sensible: two or more constraints
    combos.loc[(combos["feasible_type"] == "none") & (combos["n_constraints"] < 2),
               "feasible_type"] = None
    return combos


def _soft_weights(pattern, rng):
    if len(pattern) == 1:
        return {pattern[0]: 1.0}
    a = float(rng.choice([0.5, 0.6, 0.7]))
    first, second = (pattern if rng.random() < 0.5 else pattern[::-1])
    return {first: a, second: round(1 - a, 2)}


def sample_scenarios(combos, cases, n, split, rng,
                     type_shares=(("many", 0.45), ("few", 0.45), ("none", 0.10)),
                     exclude_users=()):
    """Specifications for n scenarios of one split, one evaluation case (user) each.

    Validation scenarios never use the held-out preference pairs or the held-out
    course, so test scenarios can measure generalisation to unseen combinations.
    Users in `exclude_users` (e.g. the validation users) are never sampled, so test
    scenarios come from unseen users.
    """
    is_val = split != "test"
    pool = combos[combos["feasible_type"].notna()]
    if is_val:
        pool = pool[pool["course"] != HELDOUT_COURSE]
    patterns = {t: [p for p in ps if not (is_val and p in HELDOUT_SOFT)]
                for t, ps in SOFT_PATTERNS.items()}
    # one case per user, in random order
    cases = cases[~cases["user_id"].isin(set(exclude_users))]
    users = cases.groupby("user_id", sort=False).sample(1, random_state=int(rng.integers(1 << 31)))
    users = users.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    if len(users) < n:
        raise ValueError(f"only {len(users)} users in {split}, {n} scenarios requested")
    types = [t for t, s in type_shares for _ in range(int(round(s * n)))]
    types = (types + ["many"] * n)[:n]
    rows = []
    for i, ftype in enumerate(types):
        choice = pool[pool["feasible_type"] == ftype].sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
        stype = ["single", "aligned", "conflicting"][i % 3]
        pattern = patterns[stype][int(rng.integers(len(patterns[stype])))]
        case = users.iloc[i]
        hard = {k: (None if pd.isna(choice[k]) else (int(choice[k]) if k == "max_time" else choice[k]))
                for k in ("diet", "course", "cuisine", "max_time")}
        rows.append({
            "scenario_id": f"{split}-{i:04d}", "split": split,
            "case_id": int(case["case_id"]), "user_id": int(case["user_id"]),
            "case_time": str(case["date"]),
            "hard": hard, "soft": _soft_weights(pattern, rng),
            "feasible_type": ftype, "soft_type": stype,
            "heldout_soft": tuple(pattern) in HELDOUT_SOFT,
            "heldout_course": hard["course"] == HELDOUT_COURSE,
        })
    return rows


def complete_scenario(spec, attrs, pref, k=10):
    """Add the feasible recipes, the number of unknown recipes and the reference list."""
    hard = {key: v for key, v in spec["hard"].items() if v is not None}
    status = hard_status(attrs, hard)
    feasible = (status == 1).all(axis=1)
    unknown = ~feasible & ~(status == 0).any(axis=1)
    util = item_utility(pref, spec["soft"])
    best = ideal_list(util, feasible, k)
    out = dict(spec)
    out.update({
        "candidates": "catalog",                     # every catalog item I(tau) is a candidate
        "n_feasible": int(feasible.sum()),
        "n_unknown": int(unknown.sum()),
        "feasible": [int(i) for i in attrs["item_idx"].to_numpy()[feasible]],
        "abstain": bool(feasible.sum() == 0),        # correct outcome when nothing is feasible
        "ideal_list": [int(i) for i in attrs["item_idx"].to_numpy()[best]],
        "ideal_list_utility": list_utility(best, util, feasible, k) if len(best) else 0.0,
    })
    return out
