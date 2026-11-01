"""GiffCF (Zhu et al., SIGIR 2024) re-implemented in PyTorch: the second backbone.

GiffCF replaces Gaussian noise by smoothing on an item-item graph (a discretised
heat equation). With the normalised user-item matrix R = D_u^{-1/4} X D_i^{-1/2}
and lambda_max the largest eigenvalue of R^T R:

    smooth(x)     = x R^T R / lambda_max
    filter(x, t)  = x + alpha * (t / T) * (smooth(x) - x)        (t = 0 ... T)

Inference starts from the smoothed history z_T = filter(c, T) and, for t = T ... 1,
predicts the clean vector x_hat = denoiser(z_t, c, smooth(c), t), then moves to
step t - 1. The trajectory is x_hat at t = T, ..., 1 followed by the final output.

This module follows the equations of the paper and the authors' TensorFlow code
(github.com/HasiNed/GiffCF, commit 3f67e02); it does not copy that code.
"""

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import svds
from torch import nn


def _sparse_tensor(m):
    m = m.tocoo()
    idx = torch.tensor(np.vstack([m.row, m.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(m.data, dtype=torch.float32), m.shape).coalesce()


def graph_operator(train):
    """R = D_u^{-1/4} X D_i^{-1/2} and the largest eigenvalue of R^T R."""
    X = sp.csr_matrix(train, dtype=np.float64)
    du = np.asarray(X.sum(1)).ravel()
    di = np.asarray(X.sum(0)).ravel()
    du[du == 0] = 1.0
    di[di == 0] = 1.0
    R = sp.diags(du ** -0.25) @ X @ sp.diags(di ** -0.5)
    s = svds(R, k=1, return_singular_vectors=False)
    return R.astype(np.float32).tocsr(), float(s[0] ** 2)


class _Timestep(nn.Module):
    """Fixed sinusoidal embedding of the step index (as in the authors' code)."""

    def __init__(self, dim, n_steps, max_wavelength=10000.0):
        super().__init__()
        scales = max_wavelength ** (-np.arange(0, dim, 2) / dim)
        angles = np.arange(n_steps + 1)[:, None] * scales[None, :]
        self.register_buffer("table", torch.tensor(
            np.concatenate([np.sin(angles), np.cos(angles)], -1), dtype=torch.float32))

    def forward(self, t):
        return self.table[t.reshape(-1)]


class _TimeEmbed(nn.Module):
    def __init__(self, hidden, n_steps):
        super().__init__()
        self.step = _Timestep(hidden, n_steps)
        self.hidden = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, t):
        return self.out(nn.functional.silu(self.hidden(self.step(t))))


class _Mixer(nn.Module):
    """Per-position mixing of a few aligned signals: Dense(2, swish) then Dense(1)."""

    def __init__(self, n_inputs, hidden=2):
        super().__init__()
        self.hidden = nn.Linear(n_inputs, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, inputs):
        x = torch.stack(inputs, dim=-1)
        return self.out(nn.functional.silu(self.hidden(x))).squeeze(-1)


def _glorot(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


class Denoiser(nn.Module):
    def __init__(self, n_items, embed_dim, n_steps):
        super().__init__()
        self.n_items, self.embed_dim = n_items, embed_dim
        self.item_embed = nn.Parameter(torch.empty(n_items, embed_dim))
        nn.init.xavier_uniform_(self.item_embed)
        self.t_embed1 = _TimeEmbed(20, n_steps)
        self.t_embed2 = _TimeEmbed(20, n_steps)
        self.embed_mixer = _Mixer(3)
        self.score_mixer = _Mixer(4)
        _glorot(self)
        nn.init.xavier_uniform_(self.item_embed)

    def forward(self, z_t, c, Ac, t):
        b = z_t.shape[0]
        te1 = self.t_embed1(t).expand(b, self.embed_dim)
        te2 = self.t_embed2(t).expand(b, self.n_items)
        z_embed = (z_t / self.n_items) @ self.item_embed
        c_norm = c.abs().sum(-1, keepdim=True).clamp_min(1.0)        # L1 norm, at least 1
        c_embed = (c / c_norm) @ self.item_embed
        x_embed = self.embed_mixer([z_embed, c_embed, te1])
        x_mid = x_embed @ self.item_embed.T
        return self.score_mixer([x_mid, c, Ac, te2])


class GiffCF(nn.Module):
    def __init__(self, train, embed_dim=200, T=3, alpha=1.5, noise_decay=1.0, dropout=0.5):
        super().__init__()
        R, lam = graph_operator(train)
        self.register_buffer("R", _sparse_tensor(R))
        self.register_buffer("Rt", _sparse_tensor(R.T.tocsr()))
        self.lam, self.T, self.alpha = lam, T, alpha
        self.noise_decay, self.dropout = noise_decay, dropout
        self.n_items = train.shape[1]
        self.denoiser = Denoiser(self.n_items, embed_dim, T)

    def smooth(self, x):
        y = torch.sparse.mm(self.R, x.T)               # users x batch
        y = torch.sparse.mm(self.Rt, y)                # items x batch
        return y.T / self.lam

    def filter(self, x, Ax, t):
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
        if t.dim() == 1:
            t = t[:, None]
        return x + self.alpha * t / self.T * (Ax - x)

    def training_loss(self, x):
        """Squared error of the clean-vector prediction at a random step (paper eq.)."""
        t = torch.randint(1, self.T + 1, (x.shape[0],), device=x.device)
        z_t = self.filter(x, self.smooth(x), t)
        c = nn.functional.dropout(x, self.dropout, training=True)
        x_pred = self.denoiser(z_t, c, self.smooth(c), t[:, None])
        return ((x - x_pred) ** 2).sum(1).mean()

    @torch.no_grad()
    def trajectory(self, x):
        """Clean guesses at t = T, ..., 1 and the final output: (T + 1, batch, n_items)."""
        Ax = self.smooth(x)
        z_t = self.filter(x, Ax, self.T)
        out = []
        for t in range(self.T, 0, -1):
            s = t - 1
            tt = torch.full((x.shape[0], 1), t, dtype=torch.long, device=x.device)
            x_pred = self.denoiser(z_t, x, Ax, tt)
            out.append(x_pred)
            Ax_pred = self.smooth(x_pred)
            z_s = self.filter(x_pred, Ax_pred, s)
            if self.noise_decay > 0:
                z_t = z_s + self.noise_decay ** (t - s) * (z_t - self.filter(x_pred, Ax_pred, t))
            else:
                z_t = z_s
        out.append(z_t)
        return torch.stack(out)

    @torch.no_grad()
    def recommend(self, x):
        return self.trajectory(x)[-1]

    def checkpoints(self):
        """Description of every logged checkpoint (smoothing level instead of noise level)."""
        rows = [{"checkpoint": k, "t": t, "smoothing": self.alpha * t / self.T,
                 "normalized_t": t / self.T} for k, t in enumerate(range(self.T, 0, -1))]
        rows.append({"checkpoint": self.T, "t": 0, "smoothing": 0.0, "normalized_t": 0.0,
                     "note": "final output"})
        return rows


# ---------------------------------------------------------------- training
def fit(model, train, evaluate, epochs=1000, batch_size=100, lr=1e-3, val_every=10,
        patience=50, seed=0, device="cuda", log=print, on_best=None):
    """Train with Adam on the users of the training matrix; keep the best state.

    evaluate(model) -> validation score (higher is better), called every `val_every`
    epochs; training stops after `patience` epochs without improvement (as in the
    authors' protocol). Returns (best score, best epoch, history).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(train, dtype=np.float32)
    opt = torch.optim.Adam(model.denoiser.parameters(), lr=lr)
    best, best_epoch, best_state, history = -np.inf, 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(X.shape[0])
        total = 0.0
        for start in range(0, len(order), batch_size):
            x = torch.from_numpy(X[order[start:start + batch_size]].toarray()).to(device)
            loss = model.training_loss(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.shape[0]
        row = {"epoch": epoch, "loss": total / X.shape[0]}
        if epoch % val_every == 0:
            model.eval()
            score = evaluate(model)
            row["val"] = score
            if score > best:
                best, best_epoch = score, epoch
                best_state = {k: v.detach().clone() for k, v in model.denoiser.state_dict().items()}
                if on_best is not None:
                    on_best(model, score, epoch)
            log(f"epoch {epoch}: loss {row['loss']:.4f} val {score:.4f} (best {best:.4f} at {best_epoch})")
            if epoch - best_epoch >= patience:
                history.append(row)
                break
        history.append(row)
    model.denoiser.load_state_dict(best_state)
    model.eval()
    return best, best_epoch, history


HYPER = {"embed_dim": 200, "T": 3, "alpha": 1.5, "noise_decay": 1.0, "dropout": 0.5}


def build(train, hyper, device):
    h = dict(HYPER, **(hyper or {}))
    return GiffCF(train, h["embed_dim"], h["T"], h["alpha"], h["noise_decay"], h["dropout"]).to(device)


def load_run(run_dir, train, device):
    """Rebuild a trained GiffCF from <run_dir>/config.json and best.pt (denoiser weights)."""
    import json
    from pathlib import Path
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    model = build(train, cfg["model"], device)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.denoiser.load_state_dict(state)
    model.eval()
    return model, cfg
