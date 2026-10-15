import numpy as np
import torch

from trajrisk.evaluate import metrics_from_ranks, target_ranks


def test_rank_ignores_seen_and_breaks_ties_by_index():
    scores = torch.tensor([[0.9, 0.5, 0.5, 0.1]])
    seen = torch.tensor([[True, False, False, False]])   # item 0 is removed
    assert target_ranks(scores, torch.tensor([2]), seen).tolist() == [1]
    assert target_ranks(scores, torch.tensor([1]), seen).tolist() == [0]


def test_metrics_for_known_ranks():
    m = metrics_from_ranks([0, 1, 50], ks=[10])
    assert np.isclose(m["recall@10"], 2 / 3)
    assert np.isclose(m["ndcg@10"], (1 + 1 / np.log2(3)) / 3)
