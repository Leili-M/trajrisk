
"""Record what DiffRec 'thinks' at every reverse step.



DiffRec turns a user's history into scores in a few reverse steps.

At each step the model makes a guess of the clean preference vector

(pred_xstart). The list of these guesses is the trajectory.

"""



import torch





@torch.no_grad()

def diffrec_trajectory(model, diffusion, history):

    """Return the clean guess at every reverse step, noisiest step first.



    Follows GaussianDiffusion.p_sample with steps=0 and no sampling noise,

    so the last guess equals the final recommendation scores.

    Shape: (n_steps, batch, n_items).

    """

    x_t = history

    guesses = []

    for i in reversed(range(diffusion.steps)):

        t = torch.full((x_t.shape[0],), i, dtype=torch.long, device=x_t.device)

        if diffusion.noise_scale == 0:

            x_t = model(x_t, t)

            guesses.append(x_t)

        else:

            out = diffusion.p_mean_variance(model, x_t, t)

            guesses.append(out["pred_xstart"])

            x_t = out["mean"]

    return torch.stack(guesses)





def topk_lists(scores, seen, k):

    """Top-k items per row after hiding items the user has already seen."""

    return scores.masked_fill(seen, float("-inf")).topk(k, dim=-1).indices





def overlap_at_k(a, b):

    """Share of items two top-k lists have in common (0 = none, 1 = same)."""

    k = a.shape[-1]

    return (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().sum(-1) / k

