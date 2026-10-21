"""Uncertainty from masked histories.

Idea (from masked diffusion language models): hide a random part of the user's
history several times and see how much the recommendation changes. If the list
stays the same whatever we hide, the answer rests on a clear taste; if it changes
a lot, it depends on a few particular items.

The list that is judged is always the normal DiffRec list from the full history.
Masking is only used to measure how stable that list is. The model and the
official DiffRec code are not changed.
"""

import torch

from trajrisk.diffrec_model import recommend_scores
from trajrisk.trajectory import overlap_at_k, topk_lists


def mask_history(history, rate):
    """Hide each liked item with probability `rate`.

    A row never becomes empty: if every item was hidden, the full row is kept.
    """
    keep = (torch.rand_like(history) >= rate).float()
    masked = history * keep
    empty = masked.sum(dim=-1) == 0
    masked[empty] = history[empty]
    return masked


@torch.no_grad()
def masked_signals(model, diffusion, history, seen, targets, n_masks, rate):
    """Uncertainty signals for one batch of cases. Returns 1-D CPU tensors."""
    full = recommend_scores(model, diffusion, history)
    top10 = topk_lists(full, seen, 10)
    top50 = topk_lists(full, seen, 50)
    top11 = full.masked_fill(seen, float("-inf")).topk(11, dim=-1).values

    change10, change50, score_change, kept = [], [], [], []
    for _ in range(n_masks):
        s = recommend_scores(model, diffusion, mask_history(history, rate))
        m10 = topk_lists(s, seen, 10)
        change10.append(1 - overlap_at_k(m10, top10))
        change50.append(1 - overlap_at_k(topk_lists(s, seen, 50), top50))
        score_change.append((s - full).norm(dim=-1) / (full.norm(dim=-1) + 1e-8))
        kept.append((top10.unsqueeze(-1) == m10.unsqueeze(-2)).any(-1))  # (batch, 10)

    always_kept = torch.stack(kept).all(dim=0).float().mean(dim=-1)
    return {
        "mask_change@10": torch.stack(change10).mean(0).cpu(),
        "mask_change@50": torch.stack(change50).mean(0).cpu(),
        "mask_score_change": torch.stack(score_change).mean(0).cpu(),
        "mask_unstable@10": (1 - always_kept).cpu(),
        "small_margin": (-(top11[:, 9] - top11[:, 10])).cpu(),
        "miss": (~(top10 == targets[:, None]).any(-1)).float().cpu(),
    }