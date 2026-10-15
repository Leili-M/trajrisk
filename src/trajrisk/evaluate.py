"""Ranking metrics when each case has one hidden target."""

import numpy as np
import torch

from .diffrec_data import iterate_batches


def target_ranks(scores, targets, seen):
    """0-based rank of each target after removing seen recipes.

    Ties are broken by recipe index (lower index first).
    """
    scores = scores.masked_fill(seen, float("-inf"))
    target_score = scores.gather(1, targets[:, None])
    greater = (scores > target_score).sum(dim=1)
    item_idx = torch.arange(scores.shape[1], device=scores.device)
    ties_before = ((scores == target_score) & (item_idx[None, :] < targets[:, None])).sum(dim=1)
    return greater + ties_before


def metrics_from_ranks(ranks, ks):
    ranks = np.asarray(ranks)
    out = {}
    for k in ks:
        hit = ranks < k
        out[f"recall@{k}"] = float(hit.mean())
        out[f"ndcg@{k}"] = float((hit / np.log2(ranks + 2)).mean())
    return out


@torch.no_grad()
def rank_cases(score_fn, history, seen, targets, device, batch_size=400):
    """Rank of the target for every case. score_fn maps a dense batch to scores."""
    ranks = []
    for idx, h in iterate_batches(history, batch_size):
        s = torch.from_numpy(seen[idx].toarray()).to(device) > 0
        t = torch.as_tensor(targets[idx], device=device, dtype=torch.long)
        ranks.append(target_ranks(score_fn(h.to(device)), t, s).cpu().numpy())
    return np.concatenate(ranks)
