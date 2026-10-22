"""RQ2.4: semantic movement of DiffRec's top-10 (needs scripts/embed_items.py first).

Counting signals ask "how many of the 10 items change?". Semantic signals ask
"how far does the MEANING of the list move?" (the centroid of the item text
embeddings). Swapping one edition of a game for another counts as a change but
barely moves the meaning.

For each case:
  traj_count / traj_sem : change of the top-10 along the denoising trajectory
                          (each step vs the final list), counted vs semantic
  mask_count / mask_sem : change of the top-10 when 50% of the history is hidden
  list_spread           : how varied the final top-10 is in meaning
  hist_spread           : how varied the user's history is in meaning
  hist_to_list          : how far the list's meaning is from the history's meaning

    sbatch jobs/run.sh configs/amazon_video_games.yaml scripts/semantic_experiment.py --run-name vg_base
Results go to <output_dir>/semantic/<run-name>{_split}/
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from trajrisk import diffrec_data as dd
from trajrisk import risk_eval
from trajrisk.config import load_config
from trajrisk.diffrec_model import build_diffrec, recommend_scores
from trajrisk.masking import mask_history
from trajrisk.trajectory import diffrec_trajectory, overlap_at_k, topk_lists

COUNT = ["traj_count", "mask_count"]
SEM = ["traj_sem", "mask_sem", "list_spread", "hist_spread", "hist_to_list"]


def step_scores(traj, batch, n_items):
    """The per-step score matrices (batch x items) from diffrec_trajectory's output."""
    if isinstance(traj, dict):
        traj = list(traj.values())
    if torch.is_tensor(traj):
        traj = list(traj) if traj.dim() == 3 else [traj]
    steps = [t for t in traj if torch.is_tensor(t) and t.shape == (batch, n_items)]
    if not steps:  # e.g. a tuple whose first element is the list of steps
        for t in traj:
            if isinstance(t, (list, tuple)) or (torch.is_tensor(t) and t.dim() == 3):
                return step_scores(t, batch, n_items)
        raise ValueError(f"Cannot read trajectory output of type {type(traj)}")
    return steps


def centroid(emb, lists):
    """Normalised mean embedding of each top-k list. lists: (batch, k) item indices."""
    c = emb[lists].mean(dim=1)
    return torch.nn.functional.normalize(c, dim=-1)


def sem_change(a, b):
    return 1 - (a * b).sum(-1)  # 1 - cosine


@torch.no_grad()
def batch_signals(model, diffusion, h, seen, targets, emb, n_masks, rate):
    n, n_items = h.shape
    full = recommend_scores(model, diffusion, h)
    top10 = topk_lists(full, seen, 10)
    c_full = centroid(emb, top10)

    # trajectory: each step's list vs the final list
    steps = step_scores(diffrec_trajectory(model, diffusion, h), n, n_items)
    tc, ts = [], []
    for s in steps:
        l = topk_lists(s, seen, 10)
        tc.append(1 - overlap_at_k(l, top10))
        ts.append(sem_change(centroid(emb, l), c_full))

    # masking: masked-history list vs the full-history list
    mc, ms = [], []
    for _ in range(n_masks):
        l = topk_lists(recommend_scores(model, diffusion, mask_history(h, rate)), seen, 10)
        mc.append(1 - overlap_at_k(l, top10))
        ms.append(sem_change(centroid(emb, l), c_full))

    # static meaning of the list and the history
    e10 = emb[top10]                                   # (n, 10, d)
    pair = (e10 @ e10.transpose(1, 2)).sum((1, 2)) - 10
    list_spread = 1 - pair / 90
    w = (h > 0).float()
    h_mean = (w @ emb) / w.sum(-1, keepdim=True).clamp(min=1)
    hist_spread = 1 - h_mean.norm(dim=-1)              # 0 = all history items alike
    hist_to_list = sem_change(torch.nn.functional.normalize(h_mean, dim=-1), c_full)

    top11 = full.masked_fill(seen, float("-inf")).topk(11, dim=-1).values
    out = {
        "traj_count": torch.stack(tc).mean(0), "traj_sem": torch.stack(ts).mean(0),
        "mask_count": torch.stack(mc).mean(0), "mask_sem": torch.stack(ms).mean(0),
        "list_spread": list_spread, "hist_spread": hist_spread, "hist_to_list": hist_to_list,
        "small_margin": -(top11[:, 9] - top11[:, 10]),
        "miss": (~(top10 == targets[:, None]).any(-1)).float(),
    }
    return {k: v.float().cpu() for k, v in out.items()}


def cv_auc(df, cols, users):
    return roc_auc_score(df["miss"], risk_eval.cv_predict(df, cols, users))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="valid_model")
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--masks", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--max-cases", type=int, default=None, help="for a quick test")
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["project"]["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(cfg["paths"]["output_dir"])

    run_dir = out_root / "models" / args.run_name
    with open(run_dir / "config.json") as f:
        cfg["model"] = json.load(f)["model"]

    items, inter, cases = dd.load_processed(cfg)
    n_items = len(items)
    emb_path = Path(cfg["paths"]["data_dir"]) / "processed" / "item_emb.npy"
    emb = torch.from_numpy(np.load(emb_path)).float().to(device)
    emb = torch.nn.functional.normalize(emb, dim=-1)
    assert emb.shape[0] == n_items, f"{emb.shape[0]} embeddings for {n_items} items"

    model, diffusion = build_diffrec(cfg, n_items, device)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    part = cases[cases["split"] == args.split].reset_index(drop=True)
    if args.max_cases:
        part = part.sample(min(args.max_cases, len(part)), random_state=seed).reset_index(drop=True)
    hist, seen = dd.case_matrices(part, inter, n_items)
    targets = part["target_idx"].to_numpy()
    users = part["user_id"].to_numpy()
    print(f"Run {args.run_name}: {len(part):,} {args.split} cases, mask rate {args.rate:.0%}, "
          f"{args.masks} masks, device {device}")

    torch.manual_seed(seed)
    t0 = time.time()
    collected, pop = [], torch.zeros(n_items)
    for idx, h in dd.iterate_batches(hist, args.batch_size):
        s = torch.from_numpy(seen[idx].toarray()).to(device) > 0
        t = torch.as_tensor(targets[idx], device=device, dtype=torch.long)
        h = h.to(device)
        collected.append(batch_signals(model, diffusion, h, s, t, emb, args.masks, args.rate))
        pop += (h > 0).float().sum(0).cpu()
    df = pd.DataFrame({k: torch.cat([c[k] for c in collected]).numpy() for k in collected[0]})
    df["miss"] = df["miss"] > 0.5
    df["history_size"] = part["n_prior"].to_numpy()
    df["user_id"] = users

    # controls (history popularity here = how often each item appears in these histories)
    df["log_history"] = np.log1p(df["history_size"])
    logpop = torch.log1p(pop)
    hp = []
    for idx, h in dd.iterate_batches(hist, args.batch_size):
        w = (h > 0).float()
        hp.append((w @ logpop) / w.sum(-1).clamp(min=1))
    df["history_popularity"] = torch.cat(hp).numpy()
    controls = ["log_history", "small_margin", "history_popularity"]

    print(f"\n({time.time() - t0:.0f} s)  Recall@10: {1 - df['miss'].mean():.4f}")
    print("\n1) Each signal alone (AUROC; 0.5 = chance, below 0.5 = higher value means fewer misses):")
    for c in COUNT + SEM:
        print(f"   {c:14s} {roc_auc_score(df['miss'], df[c]):.3f}   "
              f"corr with history length {df[c].corr(df['log_history'], method='spearman'):+.2f}")

    base = cv_auc(df, controls, users)
    cnt = cv_auc(df, controls + COUNT, users)
    print("\n2) Cross-validated AUROC (5 folds, split by user):")
    print(f"   controls only                      {base:.3f}")
    print(f"   + counting signals                 {cnt:.3f}   ({cnt - base:+.3f})")
    for c in SEM:
        a = cv_auc(df, controls + COUNT + [c], users)
        print(f"   + counting + {c:20s}  {a:.3f}   ({a - cnt:+.3f} over counting)")
    allsem = cv_auc(df, controls + COUNT + SEM, users)
    print(f"   + counting + all semantic          {allsem:.3f}   ({allsem - cnt:+.3f} over counting)")
    semonly = cv_auc(df, controls + SEM, users)
    print(f"   + semantic only (no counting)      {semonly:.3f}   ({semonly - base:+.3f} over controls)")

    split_tag = "" if args.split == "valid_model" else f"_{args.split}"
    out = out_root / "semantic" / f"{args.run_name}{split_tag}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "signals.parquet")
    with open(out / "summary.json", "w") as f:
        json.dump({"run": args.run_name, "split": args.split, "cases": len(df),
                   "recall@10": float(1 - df["miss"].mean()), "cv_controls": base,
                   "cv_counting": cnt, "cv_counting_semantic": allsem,
                   "cv_semantic_only": semonly}, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
