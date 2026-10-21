"""Simple recommenders used as reference points.

Every method returns a function score_fn(history) -> scores, where history is a
dense (batch, n_items) tensor of the user's earlier liked items. This is the
same interface as DiffRec, so evaluate.rank_cases can score all of them.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def item_counts(train):
    """How many training users liked each item."""
    return np.asarray(train.sum(axis=0)).ravel().astype(np.float32)


def popularity(train, device):
    """The same list for everyone: the most liked items in training."""
    pop = torch.tensor(item_counts(train), device=device)
    return lambda h: pop.expand(h.shape[0], -1)


def recent_popularity(inter, n_items, days, device):
    """Like popularity, but only counts the last `days` days of training."""
    tr = inter[(inter["split"] == "train") & inter["positive"]]
    recent = tr[tr["date"] > tr["date"].max() - pd.Timedelta(days=days)]
    counts = np.bincount(recent["item_idx"], minlength=n_items).astype(np.float32)
    pop = torch.tensor(counts, device=device)
    return lambda h: pop.expand(h.shape[0], -1)


def _via_users(train, device, row_weight=None, col_weight=None):
    """Item-to-item scores computed through the users who liked the items.

    score(j) = sum over history items i of row_weight[i] * C[i, j] * col_weight[j],
    where C[i, j] is the number of training users who liked both i and j.
    We compute (H * row_weight) X^T X * col_weight, so the n_items x n_items
    matrix C is never built (for HUMMUS it would need about 7 GB).
    """
    X = train.astype(np.float32).tocsr()
    XT = X.T.tocsr()

    def score(h):
        H = sp.csr_matrix(h.cpu().numpy())
        if row_weight is not None:
            H = H.multiply(row_weight).tocsr()
        S = ((H @ XT) @ X).toarray()
        if col_weight is not None:
            S *= col_weight
        return torch.tensor(S, dtype=torch.float32, device=device)

    return score


def co_occurrence(train, device):
    """People who liked your items also liked these."""
    return _via_users(train, device)


def conditional(train, device):
    """Co-occurrence, with each history item scaled down by its popularity: P(j | i)."""
    inv = 1.0 / np.maximum(item_counts(train), 1)
    return _via_users(train, device, row_weight=inv)


def item_knn(train, device):
    """Cosine similarity between items: co-occurrence divided by both items' popularity."""
    inv_sqrt = 1.0 / np.sqrt(np.maximum(item_counts(train), 1))
    return _via_users(train, device, row_weight=inv_sqrt, col_weight=inv_sqrt)


def all_methods(train, inter, n_items, device, recent_days=365):
    return {
        "popularity": popularity(train, device),
        "recent popularity": recent_popularity(inter, n_items, recent_days, device),
        "co-occurrence": co_occurrence(train, device),
        "conditional": conditional(train, device),
        "itemknn": item_knn(train, device),
    }