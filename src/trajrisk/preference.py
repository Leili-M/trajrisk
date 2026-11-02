"""Preference-aware trajectory comparison (proposal section 6, RQ2).

Everything here is plain numpy, so it can be tested without a model. Rules for
feasibility, item utility and regret come from benchmark.py (proposal: the same
definitions for scenario building, the reranker and the evaluation).

Notation: P is the (items x 5) matrix of preference percentiles (benchmark.preference_scores,
1 = best, NaN = unknown); w(q, d) are the soft-preference weights of request q;
v_q(i) the item utility; c_q(i) the number of hard constraints item i is known to violate.

Attribute distance between two recipes (proposal):
    d_q(i, j) = sum_d w(q, d) |P[i, d] - P[j, d]|     (a missing value counts as 0.5)
List distance D_q(L, L') = mean cost of the minimum-cost matching (Hungarian) between
the two lists; items left unmatched (lists of different length) cost 1.
Hard-constraint violations are kept as separate features, never averaged into d_q.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import benchmark as bm
from .signals import rank_change

PREF_NAMES = list(bm.PREFERENCES)                 # time, ingredients, calories, protein, health
K = 10
FAIL_REGRET = 0.5                                 # prespecified failure: regret >= 0.5


# ---------------------------------------------------------------- request-aware reranker
def zscore(scores, seen):
    """Standardise one score vector over the items the user has not seen."""
    x = scores[~seen]
    sd = x.std()
    return (scores - x.mean()) / (sd if sd > 0 else 1.0)


def rerank(z, util, viol, beta, gamma, seen, feasible=None, k=K):
    """Proposal formula 9: normalised model score + beta * soft utility - gamma * violations.

    feasible is None: soft enforcement (violations stay possible).
    feasible given:   hard filtering (only known-feasible recipes; may return < k items).
    Returns the top-k item indices, padded with -1.
    """
    s = z + beta * np.nan_to_num(util, nan=0.0) - gamma * viol
    s = np.where(seen, -np.inf, s)
    if feasible is not None:
        s = np.where(feasible, s, -np.inf)
    n_ok = int(np.isfinite(s).sum())
    m = min(k, n_ok)
    if m == 0:
        return np.full(k, -1, dtype=np.int64)
    top = np.argpartition(-s, m - 1)[:m]
    top = top[np.argsort(-s[top], kind="stable")]
    return np.concatenate([top, np.full(k - m, -1)]).astype(np.int64)


# ---------------------------------------------------------------- list distances
def _items(lst):
    return lst[lst >= 0]


def pair_costs(P, w, a, b):
    """d_q between every item of list a and every item of list b (len(a) x len(b))."""
    diff = np.abs(P[a][:, None, :] - P[b][None, :, :])
    diff = np.where(np.isnan(diff), 0.5, diff)
    diff[a[:, None] == b[None, :]] = 0.0              # the same recipe is at distance 0
    return diff @ w


def matching_distance(cost, n_a, n_b):
    """Mean cost of the min-cost matching; unmatched items cost 1."""
    n = max(n_a, n_b)
    if n == 0:
        return 0.0
    if n_a == 0 or n_b == 0:
        return 1.0
    r, c = linear_sum_assignment(cost)
    return float((cost[r, c].sum() + abs(n_a - n_b)) / n)


def attr_distance(P, w, la, lb):
    a, b = _items(la), _items(lb)
    return matching_distance(pair_costs(P, w, a, b) if len(a) and len(b) else np.zeros((0, 0)),
                             len(a), len(b))


def util_distance(util, la, lb):
    """Proposal: utility distance |v_q(i) - v_q(j)|, matched like the attribute distance."""
    a, b = _items(la), _items(lb)
    if not len(a) or not len(b):
        return matching_distance(None, len(a), len(b))
    cost = np.abs(util[a][:, None] - util[b][None, :])
    cost = np.where(np.isnan(cost), 0.5, cost)
    cost[a[:, None] == b[None, :]] = 0.0
    return matching_distance(cost, len(a), len(b))


def rank_aligned_distance(P, w, la, lb):
    """Ablation when position matters: compare item at rank r with item at rank r,
    discounted by 1 / log2(r + 2). A missing item on one side costs 1."""
    disc = 1 / np.log2(np.arange(len(la)) + 2)
    cost = np.ones(len(la))
    both = (la >= 0) & (lb >= 0)
    if both.any():
        d = np.abs(P[la[both]] - P[lb[both]])
        d = np.where(np.isnan(d), 0.5, d)
        d[la[both] == lb[both]] = 0.0
        cost[both] = d @ w
    cost[(la < 0) & (lb < 0)] = 0.0
    return float((disc * cost).sum() / disc.sum())


def distribution_distance(P, w, la, lb):
    """Cheaper comparison: weighted difference of the lists' mean preference percentiles."""
    a, b = _items(la), _items(lb)
    if not len(a) or not len(b):
        return 1.0 if len(a) != len(b) else 0.0
    with np.errstate(all="ignore"):
        ma, mb = np.nanmean(P[a], axis=0), np.nanmean(P[b], axis=0)
    d = np.abs(ma - mb)
    d = np.where(np.isnan(ma) & np.isnan(mb), 0.0, np.where(np.isnan(d), 0.5, d))
    return float(d @ w)


def _padded_for_rank(lst, offset):
    """Give padding slots unique negative ids so that empty slots never match."""
    out = lst.copy()
    pad = out < 0
    out[pad] = -(offset + np.flatnonzero(pad) + 1)
    return out


# ---------------------------------------------------------------- per-scenario quantities
def scenario_signals(lists, P, w, util, viol, unknown, k=K):
    """All section-6 signals for one scenario. lists: (C, k) item ids per checkpoint
    (last = final), padded with -1. No outcome enters these computations."""
    final = lists[-1]
    nonfinal = lists[:-1]
    w_generic = np.full(len(w), 1.0 / len(w))
    bad = lambda l: (viol[_items(l)] > 0).mean() if len(_items(l)) else 0.0
    out = {
        "u_set": np.mean([1 - len(np.intersect1d(_items(l), _items(final))) / k for l in nonfinal]),
        "u_rank": float(np.mean(rank_change(
            np.stack([_padded_for_rank(l, 0) for l in nonfinal]),
            np.stack([_padded_for_rank(final, 1000)] * len(nonfinal))))),
        "u_attr": np.mean([attr_distance(P, w, l, final) for l in nonfinal]),
        "u_util": np.mean([util_distance(util, l, final) for l in nonfinal]),
        "u_attr_rank": np.mean([rank_aligned_distance(P, w, l, final) for l in nonfinal]),
        "u_dist": np.mean([distribution_distance(P, w, l, final) for l in nonfinal]),
        "u_generic": np.mean([attr_distance(P, w_generic, l, final) for l in nonfinal]),
        "viol_traj": np.mean([abs(bad(l) - bad(final)) for l in nonfinal]),
        # direct checks of the final list (proposal section 7 baseline)
        "viol_final": bad(final),
        "unknown_final": unknown[_items(final)].mean() if len(_items(final)) else 0.0,
        "util_final": float(np.nanmean(util[_items(final)])) if len(_items(final)) else 0.0,
    }
    return {name: float(v) for name, v in out.items()}


def scenario_outcome(final, util, feasible, viol, k=K):
    """Outcomes used only for evaluation: regret, violations, failure, list size."""
    items = _items(final)
    reg = bm.regret(items, util, feasible, k) if len(items) else np.nan
    return {"regret": reg, "n_violating": int((viol[items] > 0).sum()),
            "list_len": int(len(items)),
            "failure": bool(reg >= FAIL_REGRET) if not np.isnan(reg) else np.nan}


def intra_list_diversity(P, final):
    """Request-agnostic diversity: mean pairwise attribute distance inside the list."""
    a = _items(final)
    if len(a) < 2:
        return np.nan
    c = pair_costs(P, np.full(P.shape[1], 1.0 / P.shape[1]), a, a)
    return float(c[np.triu_indices(len(a), 1)].mean())


SIGNALS_IDENTITY = ["u_set", "u_rank"]
SIGNALS_PREFERENCE = ["u_attr", "u_util", "u_attr_rank", "u_dist"]
SIGNALS_OTHER = ["u_generic", "viol_traj"]
FINAL_CHECKS = ["viol_final", "unknown_final", "util_final"]
