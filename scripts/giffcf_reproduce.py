"""Check the PyTorch re-implementation of GiffCF against the authors' published result.

Uses the MovieLens-1M split shipped with the authors' code (DiffRec's random split,
not our temporal split) and their training protocol: Adam 1e-3, batch 100, T = 3,
alpha = 1.5, validation recall@20 every 10 epochs, early stopping after 200 epochs.
The authors report test Recall@20 = 0.2024 and NDCG@20 = 0.1272 for this setting.

    git clone https://github.com/HasiNed/GiffCF external/GiffCF
    sbatch --partition=gpumedium --time=02:00:00 jobs/run.sh configs/movielens.yaml \
        scripts/giffcf_reproduce.py --data external/GiffCF/datasets/ml-1m_clean
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from trajrisk import giffcf as gc

REPO = Path(__file__).resolve().parents[1]


def load(path):
    parts = [np.load(Path(path) / f"{s}_list.npy") for s in ("train", "valid", "test")]
    n_users = max(p[:, 0].max() for p in parts) + 1
    n_items = max(p[:, 1].max() for p in parts) + 1
    return [sp.csr_matrix((np.ones(len(p), np.float32), (p[:, 0], p[:, 1])), shape=(n_users, n_items))
            for p in parts]


@torch.no_grad()
def metrics(model, history, target, mask, ks=(10, 20), batch=500, device="cuda"):
    """The authors' definitions: recall = hits / min(|target|, k); NDCG with ideal DCG."""
    users = np.flatnonzero(np.asarray(target.sum(1)).ravel() > 0)
    disc = 1 / np.log2(np.arange(2, max(ks) + 2))
    sums = {f"{m}@{k}": 0.0 for m in ("recall", "ndcg") for k in ks}
    for s in range(0, len(users), batch):
        u = users[s:s + batch]
        h = torch.from_numpy(history[u].toarray()).to(device)
        scores = model.recommend(h).masked_fill(torch.from_numpy(mask[u].toarray() > 0).to(device), -np.inf)
        top = scores.topk(max(ks), dim=1).indices.cpu().numpy()
        tgt = target[u].toarray() > 0
        hits = np.take_along_axis(tgt, top, 1).astype(float)
        n_t = tgt.sum(1)
        for k in ks:
            ideal = np.minimum(n_t, k)
            sums[f"recall@{k}"] += (hits[:, :k].sum(1) / ideal).sum()
            idcg = np.cumsum(disc)[ideal - 1]
            sums[f"ndcg@{k}"] += ((hits[:, :k] * disc[:k]).sum(1) / idcg).sum()
    return {k: v / len(users) for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default=None, help="unused; accepted for jobs/run.sh")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=200)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, valid, test = load(args.data)
    model = gc.build(train, {}, device)
    best, epoch, hist = gc.fit(
        model, train, lambda m: metrics(m, train, valid, train, device=device)["recall@20"],
        patience=args.patience, seed=args.seed, device=device)
    res = metrics(model, train, test, train + valid, device=device)
    paper = {"recall@10": 0.1308, "recall@20": 0.2024, "ndcg@10": 0.1002, "ndcg@20": 0.1272}
    lines = ["# GiffCF re-implementation check (MovieLens-1M, authors' split)", "",
             f"Best validation recall@20 {best:.4f} at epoch {epoch}.", "",
             "| Metric | Ours (PyTorch) | Authors (TensorFlow) |", "|---|---|---|"]
    lines += [f"| {k} | {res[k]:.4f} | {paper[k]:.4f} |" for k in paper]
    out = REPO / "reports" / "backbones"
    out.mkdir(parents=True, exist_ok=True)
    (out / "giffcf_reproduction.md").write_text("\n".join(lines) + "\n")
    (out / "giffcf_reproduction.json").write_text(json.dumps(
        {"test": res, "paper": paper, "best_val_recall@20": best, "best_epoch": epoch,
         "history": hist}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
