"""Trajectory logs (proposal section 4).

For every case and every denoising checkpoint the log keeps
  - the noise level of the checkpoint,
  - the top-M items and their scores (after hiding already seen items),
  - summary statistics of the FULL score vector that later signals need:
    distance to the final scores, the norm of the final scores and the
    score of the 10th, 11th and M-th item.
Held-out target ranks go to a separate evaluation table (labels), never into
the trajectory log, so signals computed from the log cannot see the outcome.

Every list of length K <= M is exact; only quantities beyond rank M are lost,
and those are covered by the summary statistics.
"""

import numpy as np
import torch

from .evaluate import target_ranks


def noise_schedule(diffusion):
    """Noise level of every reverse step, in the order the trajectory visits them."""
    ac = diffusion.alphas_cumprod
    ac = ac.detach().cpu().numpy() if torch.is_tensor(ac) else np.asarray(ac, dtype=float)
    steps = diffusion.steps
    rows = []
    for k, t in enumerate(reversed(range(steps))):
        rows.append({"checkpoint": k, "t": t,
                     "noise_std": float(np.sqrt(max(1.0 - ac[t], 0.0))),
                     "normalized_t": float(t / max(steps - 1, 1)) if steps > 1 else 0.0})
    return rows


@torch.no_grad()
def log_batch(traj, seen, targets, top_m):
    """traj: (n_checkpoints, batch, n_items) scores, noisiest first, last = final.

    Returns numpy arrays:
      items  (batch, n_checkpoints, M) int32     top-M items per checkpoint
      scores (batch, n_checkpoints, M) float32   their scores
      dist_to_final (batch, n_checkpoints)       ||s(k) - s*||_2 on the full vector
      final_norm (batch,)                        ||s*||_2
      and, kept apart, ranks (batch, n_checkpoints): 0-based target rank per checkpoint
    """
    n_ck, b, n = traj.shape
    m = min(top_m, n)
    masked = traj.masked_fill(seen.unsqueeze(0), float("-inf"))
    top = masked.topk(m, dim=-1)                                   # (n_ck, b, m)
    final = traj[-1]
    dist = (traj - final.unsqueeze(0)).norm(dim=-1)                # full vectors
    ranks = torch.stack([target_ranks(traj[k], targets, seen) for k in range(n_ck)])
    return {
        "items": top.indices.permute(1, 0, 2).to(torch.int32).cpu().numpy(),
        "scores": top.values.permute(1, 0, 2).float().cpu().numpy(),
        "dist_to_final": dist.T.float().cpu().numpy(),
        "final_norm": final.norm(dim=-1).float().cpu().numpy(),
    }, ranks.T.cpu().numpy()
