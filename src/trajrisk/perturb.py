"""Signals from perturbed DiffRec runs (proposal section 7 baselines and an extension).

Repeated sampling (proposal section 7: "repeated diffusion sampling at 2, 5, and 10 runs"):
  each sample uses the official noisy start and sampling noise
  (GaussianDiffusion.p_sample with sampling_noise=True).
Masked histories (extension, earlier experiments): each liked item is hidden with
  probability `rate` (masking.mask_history).

In both cases the list that is judged is the ordinary DiffRec list L* (no noise, full
history), the same list as in the trajectory logs and their labels. The perturbed runs
only measure how stable L* is: for each perturbed run, the change of its top-K list
against L* (set and rank, formulas 4 and 5) and the relative change of the score vector.
No target or label enters these functions.
"""

import torch

from .masking import mask_history
from .trajectory import topk_lists

SAMPLE_COUNTS = (2, 5, 10)


def set_change_t(a, b):
    """1 - |a ∩ b| / K for top-K lists of shape (..., K)."""
    return 1.0 - (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean(-1)


def rank_change_t(a, b):
    """Normalised absolute rank displacement (formula 5), torch version of signals.rank_change."""
    k = a.shape[-1]
    eq = a.unsqueeze(-1) == b.unsqueeze(-2)                       # (..., K, K)
    pos = torch.arange(1, k + 1, device=a.device, dtype=torch.float32)
    matched = (eq.float() * (pos[:, None] - pos[None, :]).abs()).sum((-1, -2))
    a_in_b, b_in_a = eq.any(-1), eq.any(-2)
    only_a = ((~a_in_b).float() * (k + 1 - pos)).sum(-1)
    only_b = ((~b_in_a).float() * (k + 1 - pos)).sum(-1)
    union = 2 * k - a_in_b.float().sum(-1)
    return (matched + only_a + only_b) / (k * union)


def _changes(scores, ref_scores, ref_top, seen, k):
    top = topk_lists(scores, seen, k)
    rel = (scores - ref_scores).norm(dim=-1) / (ref_scores.norm(dim=-1) + 1e-8)
    return set_change_t(top, ref_top), rank_change_t(top, ref_top), rel, top


@torch.no_grad()
def perturbation_signals(model, diffusion, history, seen, n_samples=10, n_masks=10,
                         rate=0.5, k=10):
    """Returns (signals dict of 1-D CPU tensors, final top-K list L*)."""
    ref = diffusion.p_sample(model, history, 0, False)               # ordinary DiffRec scores
    ref_top = topk_lists(ref, seen, k)
    out = {}

    s_set, s_rank, s_score = [], [], []
    for _ in range(n_samples):
        s = diffusion.p_sample(model, history, diffusion.steps, True)   # noisy start + noise
        a, b, c, _ = _changes(s, ref, ref_top, seen, k)
        s_set.append(a); s_rank.append(b); s_score.append(c)
    s_set, s_rank, s_score = torch.stack(s_set), torch.stack(s_rank), torch.stack(s_score)
    for n in SAMPLE_COUNTS:
        if n <= n_samples:
            out[f"samp{n}_set"] = s_set[:n].mean(0)
            out[f"samp{n}_rank"] = s_rank[:n].mean(0)
            out[f"samp{n}_score"] = s_score[:n].mean(0)

    m_set, m_rank, m_score, kept = [], [], [], []
    for _ in range(n_masks):
        s = diffusion.p_sample(model, mask_history(history, rate), 0, False)
        a, b, c, top = _changes(s, ref, ref_top, seen, k)
        m_set.append(a); m_rank.append(b); m_score.append(c)
        kept.append((ref_top.unsqueeze(-1) == top.unsqueeze(-2)).any(-1))
    out["mask_set"] = torch.stack(m_set).mean(0)
    out["mask_rank"] = torch.stack(m_rank).mean(0)
    out["mask_score"] = torch.stack(m_score).mean(0)
    out["mask_unstable"] = 1.0 - torch.stack(kept).all(0).float().mean(-1)
    return {name: v.float().cpu() for name, v in out.items()}, ref_top.cpu()
