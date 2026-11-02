"""Uncertainty signals computed from trajectory logs (proposal section 5).

Input is one trajectory log (scripts/log_trajectory.py): for every case and every
checkpoint the top-M items and scores (noisiest checkpoint first, last = final output),
the distance of each checkpoint's full score vector to the final one, and the norm of
the final vector. Labels are NOT an input: no function here receives the target or
its rank, so every signal is computable before the outcome is known.

All signals are oriented so that a higher value means a higher estimated risk.

Notation (proposal): L(s) top-K list at a non-final checkpoint s, L* final top-K list,
S = number of non-final checkpoints, weights w_s >= 0 with sum 1.

  U_set  (formula 4):  sum_s w_s (1 - |L(s) ∩ L*| / K)
  U_rank (formula 5):  sum_s w_s  sum_{i in A_s} |r_s(i) - r*(i)| / (K |A_s|),
                       A_s = L(s) ∪ L*, absent items get rank K + 1.

Weightings ("Compare uniform and late-checkpoint weighting"): w_s ∝ rho^(S-1-s);
rho = 1 is uniform, smaller rho puts more weight on the late (less noisy) checkpoints.
Early-only / late-only: uniform weights over the first S//2 / remaining non-final
checkpoints. Consecutive versions compare each checkpoint with the next one.
"""

import numpy as np

RHOS = (1.0, 0.75, 0.5, 0.25)          # 1.0 = uniform weighting
TAUS = (0.1, 0.3, 1.0, 3.0)            # entropy temperatures (multiples of a score scale)


def weight_tag(rho):
    return "unif" if rho == 1.0 else f"r{rho:g}"


def late_weights(n, rho):
    """Weights over n non-final checkpoints (noisiest first), larger on late ones if rho < 1."""
    w = float(rho) ** np.arange(n - 1, -1, -1, dtype=float)
    return w / w.sum()


def set_change(a, b):
    """1 - |a ∩ b| / K for two lists of item ids, shape (N, K) each."""
    eq = a[:, :, None] == b[:, None, :]
    return 1.0 - eq.any(axis=2).sum(axis=1) / a.shape[1]


def rank_change(a, b):
    """Normalised absolute rank displacement between two top-K lists (formula 5).

    Ranks are 1..K; an item missing from one list has rank K + 1 there. The sum over
    the union A is divided by K |A|.
    """
    _, k = a.shape
    eq = a[:, :, None] == b[:, None, :]                       # (N, K, K)
    pos = np.arange(1, k + 1)
    a_in_b = eq.any(axis=2)                                   # item of a found in b
    b_in_a = eq.any(axis=1)                                   # item of b found in a
    matched = (eq * np.abs(pos[:, None] - pos[None, :])).sum(axis=(1, 2))
    only_a = ((~a_in_b) * (k + 1 - pos)).sum(axis=1)
    only_b = ((~b_in_a) * (k + 1 - pos)).sum(axis=1)
    union = 2 * k - a_in_b.sum(axis=1)
    return (matched + only_a + only_b) / (k * union)


def _combine(per_step, prefix, out):
    """Weighted versions, early-only and late-only from a (N, S) matrix of per-step values."""
    n_steps = per_step.shape[1]
    for rho in RHOS:
        out[f"{prefix}_{weight_tag(rho)}"] = per_step @ late_weights(n_steps, rho)
    half = max(n_steps // 2, 1)
    out[f"{prefix}_early"] = per_step[:, :half].mean(axis=1)
    out[f"{prefix}_late"] = per_step[:, half:].mean(axis=1) if n_steps > half else per_step[:, -1]


def entropy(final_scores, tau, scale):
    """Softmax entropy over the logged top-M final scores, divided by log M.

    Temperature = tau * scale, where scale is a fixed score scale of the run (std of the
    final top-M scores on the model-validation cases), so tau means the same for every
    model. Only the top-M scores are logged, so this is a truncated entropy.
    """
    s = np.asarray(final_scores, dtype=np.float64)          # float32 logs underflow to 0
    z = (s - s[:, :1]) / (tau * scale)
    z = np.where(np.isfinite(z), z, -np.inf)                 # -inf: item not available
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    p = np.exp(log_p)
    plogp = np.where(p > 0, p * np.where(p > 0, log_p, 0.0), 0.0)   # 0 * log 0 = 0
    return -plogp.sum(axis=1) / np.log(s.shape[1])


def compute_signals(items, scores, dist_to_final, final_norm, score_scale, k=10):
    """All section-5 signals for one trajectory log. Returns a dict of (N,) arrays.

    items, scores: (N, C, M) with the last checkpoint = final output; dist_to_final: (N, C);
    final_norm: (N,); score_scale: positive float for the entropy temperature.
    """
    n_ck = items.shape[1]
    if n_ck < 2:
        raise ValueError("A trajectory needs at least one non-final checkpoint.")
    final = items[:, -1, :k]
    out = {}

    # trajectory against the final list (formulas 4 and 5)
    per_set = np.stack([set_change(items[:, s, :k], final) for s in range(n_ck - 1)], axis=1)
    per_rank = np.stack([rank_change(items[:, s, :k], final) for s in range(n_ck - 1)], axis=1)
    _combine(per_set, "u_set", out)
    _combine(per_rank, "u_rank", out)
    if items.shape[2] >= 50:                                 # only if 50 items were logged
        out["u_set50_unif"] = np.stack(
            [set_change(items[:, s, :50], items[:, -1, :50]) for s in range(n_ck - 1)], axis=1
        ).mean(axis=1)

    # consecutive checkpoints: gradual convergence vs repeated reversals
    cons_set = np.stack([set_change(items[:, s, :k], items[:, s + 1, :k])
                         for s in range(n_ck - 1)], axis=1)
    cons_rank = np.stack([rank_change(items[:, s, :k], items[:, s + 1, :k])
                          for s in range(n_ck - 1)], axis=1)
    out["cons_set"] = cons_set.mean(axis=1)
    out["cons_rank"] = cons_rank.mean(axis=1)
    # path length minus direct distance (first checkpoint -> final); 0 = no reversals
    out["reversal_set"] = np.clip(cons_set.sum(axis=1) - per_set[:, 0], 0.0, None)

    # score-level change (full vectors), relative to the norm of the final scores
    rel = dist_to_final[:, :-1] / np.maximum(final_norm[:, None], 1e-12)
    out["score_change_first"] = rel[:, 0]
    _combine(rel, "score_change", out)

    # final-output confidence (formula 6), negated so that higher = riskier
    fs = scores[:, -1, :]
    out["small_margin"] = -(fs[:, k - 1] - fs[:, k])
    out["top_margin"] = -(fs[:, 0] - fs[:, 1])
    for tau in TAUS:
        out[f"entropy_t{tau:g}"] = entropy(fs, tau, score_scale)
    out = {name: np.asarray(v, dtype=np.float64) for name, v in out.items()}
    bad = [name for name, v in out.items() if not np.isfinite(v).all()]
    if bad:
        raise ValueError(f"Non-finite values in signals: {bad}")
    return out


SIGNAL_GROUPS = {
    "U_set": ["u_set_unif", "u_set_r0.75", "u_set_r0.5", "u_set_r0.25",
              "u_set_early", "u_set_late", "u_set50_unif"],
    "U_rank": ["u_rank_unif", "u_rank_r0.75", "u_rank_r0.5", "u_rank_r0.25",
               "u_rank_early", "u_rank_late"],
    "consecutive": ["cons_set", "cons_rank", "reversal_set"],
    "score change": ["score_change_first", "score_change_unif", "score_change_r0.75",
                     "score_change_r0.5", "score_change_r0.25",
                     "score_change_early", "score_change_late"],
    "final output": ["small_margin", "top_margin"] + [f"entropy_t{t:g}" for t in TAUS],
}
CONTROLS = ["log_history", "history_popularity", "rec_popularity"]
