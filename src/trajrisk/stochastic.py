"""Noisy, random DiffRec trajectories and the uncertainty signals built from them.

Compared with the deterministic trajectory in trajectory.py (Part 10), two things change:

1. noisy_start:    the input is the history with forward noise added, exactly what the
                   official GaussianDiffusion.p_sample does when steps > 0.
2. sampling_noise: random noise is added after every reverse step, exactly the official
                   sampling_noise=True option.

Running this several times per case gives several different answers. How much those
answers disagree is a new uncertainty signal. The official DiffRec files are not changed;
only their public functions (q_sample, p_mean_variance) are called.
"""

import torch


@torch.no_grad()
def sample_trajectory(model, diffusion, history, noisy_start=True, sampling_noise=True):
    """One random trajectory: the clean guess at every reverse step, noisiest step first.

    With noisy_start=False and sampling_noise=False this is the Part 10 trajectory.
    With both True it follows diffusion.p_sample(model, history, diffusion.steps, True)
    step by step, so the last guess equals the official sampled scores.
    Shape: (n_steps, batch, n_items).
    """
    if diffusion.noise_scale == 0:
        raise ValueError("noise_scale is 0, so this model has no noise to sample from")
    batch, device = history.shape[0], history.device

    if noisy_start:
        t = torch.full((batch,), diffusion.steps - 1, dtype=torch.long, device=device)
        x_t = diffusion.q_sample(history, t)
    else:
        x_t = history

    guesses = []
    for i in reversed(range(diffusion.steps)):
        t = torch.full((batch,), i, dtype=torch.long, device=device)
        out = diffusion.p_mean_variance(model, x_t, t)
        guesses.append(out["pred_xstart"])
        if sampling_noise and i > 0:          # no noise on the last step, as in p_sample
            x_t = out["mean"] + torch.exp(0.5 * out["log_variance"]) * torch.randn_like(x_t)
        else:
            x_t = out["mean"]
    return torch.stack(guesses)


def noise_level(diffusion):
    """How much noise the forward process adds at its last step (history values are 0/1)."""
    if diffusion.noise_scale == 0:
        return {"noise_var_last_step": 0.0, "noise_std_last_step": 0.0, "signal_kept": 1.0}
    a = float(diffusion.alphas_cumprod[-1])
    return {"noise_var_last_step": 1 - a, "noise_std_last_step": (1 - a) ** 0.5,
            "signal_kept": a ** 0.5}


def _topk(scores, seen, k):
    return scores.masked_fill(seen, float("-inf")).topk(k, dim=-1).indices


def _overlap(a, b):
    """Share of items two top-k lists have in common (0 = none, 1 = same). Any leading dims."""
    return (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean(-1)


@torch.no_grad()
def case_signals(model, diffusion, history, seen, targets, n_samples,
                 noisy_start=True, sampling_noise=True):
    """Uncertainty signals for one batch of cases.

    The recommendation is the top-10 of the scores averaged over the samples
    (with one noise-free sample this is the ordinary DiffRec list).
    Returns a dict of 1-D CPU tensors, one value per case.
    """
    finals, step_change10, step_change50, score_change = [], [], [], []
    for _ in range(n_samples):
        traj = sample_trajectory(model, diffusion, history, noisy_start, sampling_noise)
        final = traj[-1]
        finals.append(final)

        # Within one trajectory: how much do the steps differ from the last step? (Part 10)
        t10 = torch.stack([_topk(g, seen, 10) for g in traj])
        t50 = torch.stack([_topk(g, seen, 50) for g in traj])
        step_change10.append(1 - _overlap(t10[:-1], t10[-1].unsqueeze(0)).mean(0))
        step_change50.append(1 - _overlap(t50[:-1], t50[-1].unsqueeze(0)).mean(0))
        score_change.append((traj[0] - final).norm(dim=-1) / (final.norm(dim=-1) + 1e-8))

    finals = torch.stack(finals)                         # (samples, batch, items)
    mean_scores = finals.mean(0)
    top10 = _topk(mean_scores, seen, 10)                 # the recommendation
    top50 = _topk(mean_scores, seen, 50)
    sample_top10 = torch.stack([_topk(f, seen, 10) for f in finals])
    sample_top50 = torch.stack([_topk(f, seen, 50) for f in finals])

    # Across samples: how much do the random answers disagree with the averaged one?
    disagree10 = 1 - _overlap(sample_top10, top10.unsqueeze(0)).mean(0)
    disagree50 = 1 - _overlap(sample_top50, top50.unsqueeze(0)).mean(0)

    # Rank instability: rank of each recommended item in every sample, spread across samples
    masked = finals.masked_fill(seen.unsqueeze(0), float("-inf"))
    ranks = masked.argsort(dim=-1, descending=True).argsort(dim=-1).float()
    rec_ranks = ranks.gather(-1, top10.unsqueeze(0).expand(n_samples, -1, -1))
    rank_spread = rec_ranks.std(0, unbiased=False).mean(-1) if n_samples > 1 \
        else torch.zeros(history.shape[0], device=history.device)

    s11 = mean_scores.masked_fill(seen, float("-inf")).topk(11, dim=-1).values
    hit = (top10 == targets[:, None]).any(-1)
    hit_one_sample = (sample_top10[0] == targets[:, None]).any(-1)

    out = {
        "step_change@10": torch.stack(step_change10).mean(0),
        "step_change@50": torch.stack(step_change50).mean(0),
        "step_score_change": torch.stack(score_change).mean(0),
        "sample_disagree@10": disagree10,
        "sample_disagree@50": disagree50,
        "sample_rank_spread": rank_spread,
        "small_margin": -(s11[:, 9] - s11[:, 10]),
        "miss": ~hit,
        "miss_one_sample": ~hit_one_sample,
    }
    return {k: v.float().cpu() for k, v in out.items()}
