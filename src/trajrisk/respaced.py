"""DiffRec trajectories with fewer sampling steps (proposal section 4).

A model trained with T diffusion steps can be sampled on a subset of its steps,
for example [4, 2, 0] instead of [4, 3, 2, 1, 0]. Following the standard
respacing of diffusion models, the posterior between two kept steps uses the
cumulative noise levels of those steps:

    abar_j = alphas_cumprod[t_j],   abar_prev = alphas_cumprod[t_{j+1}] (1 at the end)
    beta_j = 1 - abar_j / abar_prev
    mean   = beta_j sqrt(abar_prev) / (1 - abar_j) * x0_hat
           + (1 - abar_prev) sqrt(1 - beta_j) / (1 - abar_j) * x_t

The network always receives the original step t_j. With all T steps this is
exactly DiffRec's own reverse process, so the full trajectory is unchanged.
"""

import numpy as np
import torch


def step_subsets(n_steps):
    """Evenly spaced subsets of the trained steps, from all steps down to two."""
    out = {}
    for s in range(n_steps, 1, -1):
        ts = np.unique(np.round(np.linspace(n_steps - 1, 0, s)).astype(int))[::-1]
        out[len(ts)] = [int(t) for t in ts]
    return out


def _alphas_cumprod(diffusion):
    ac = diffusion.alphas_cumprod
    return ac.detach().cpu().numpy().astype(float) if torch.is_tensor(ac) else np.asarray(ac, float)


@torch.no_grad()
def respaced_trajectory(model, diffusion, history, steps):
    """Clean guesses at the kept steps (noisiest first), deterministic.

    steps: descending list of trained steps to visit, ending with 0.
    Returns (len(steps), batch, n_items); the last guess is the final recommendation.
    """
    if list(steps) != sorted(steps, reverse=True) or steps[-1] != 0:
        raise ValueError("steps must be descending and end with 0")
    ac = _alphas_cumprod(diffusion)
    x_t, guesses = history, []
    for j, t in enumerate(steps):
        t_vec = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
        x0 = model(x_t, t_vec)
        guesses.append(x0)
        abar = ac[t]
        abar_prev = ac[steps[j + 1]] if j + 1 < len(steps) else 1.0
        beta = 1.0 - abar / abar_prev
        c0 = beta * np.sqrt(abar_prev) / (1.0 - abar)
        ct = (1.0 - abar_prev) * np.sqrt(1.0 - beta) / (1.0 - abar)
        x_t = c0 * x0 + ct * x_t
    return torch.stack(guesses)


def normalized_level(t, n_steps):
    """Trained step t as a share of the noise range: 1 = noisiest step, 0 = final."""
    return t / (n_steps - 1) if n_steps > 1 else 0.0
