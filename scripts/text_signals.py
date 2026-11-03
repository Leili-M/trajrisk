"""Proposal section 7: text (MiniLM) baselines, evaluated with the main protocol.

Needs scripts/embed_items.py first (Amazon datasets only): <data_dir>/processed/item_emb.npy,
one unit-length all-MiniLM-L6-v2 vector per item, in item_idx order (title + description;
the rating fields of the metadata are never read).

Stage 1 (no labels): for every run and split, the stored trajectory log and the user's
history give four text signals per case, written to text_signals.parquet next to the log:
  traj_sem      generic semantic list similarity along the trajectory (proposal section 7):
                mean over the non-final checkpoints of 1 - cos(centroid(L(s)), centroid(L*)),
                where the centroid is the normalised mean embedding of a top-10 list
  list_spread   how varied the final top-10 is in meaning (1 - mean pairwise cosine)
  hist_to_list  how far the final top-10 is from the history in meaning
                (1 - cos(mean history embedding, centroid(L*)))
  hist_spread   how varied the history is in meaning (1 - norm of its mean embedding)

Stage 2: the same risk model as scripts/judge.py (fitted on risk-validation, applied once
to test, section-5 choices from risk-validation) with these signals added. Nothing is
chosen on the test cases.

    sbatch --partition=gpumedium --time=01:00:00 jobs/run.sh configs/amazon_video_games.yaml \
        scripts/text_signals.py --config configs/amazon_video_games.yaml \
        --runs vg_base vg_seed1 vg_seed2 vg_seed3 vg_seed4
Report: reports/text/<dataset>[_giffcf].md and .json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge  # noqa: E402  (same loaders, predictor sets and fitting as the main analysis)
from trajrisk import judge_eval as je  # noqa: E402

K = 10
SPLITS = (judge.FIT_SPLIT, judge.EVAL_SPLIT)
TEXT_STATIC = ["list_spread", "hist_to_list", "hist_spread"]
TEXT_TRAJ = ["traj_sem"]
TEXT_ALL = TEXT_STATIC + TEXT_TRAJ


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def text_signals(items, hist_mean, emb, k=K, batch=10000):
    """items: (N, C, M) top-M item ids per checkpoint (last = final); hist_mean: (N, d)
    mean history embedding; emb: (n_items, d) unit vectors. Returns a dict of (N,) arrays."""
    n, c, _ = items.shape
    out = {name: np.empty(n, dtype=np.float32) for name in TEXT_ALL}
    for a in range(0, n, batch):
        b = min(a + batch, n)
        e = emb[items[a:b, :, :k]]                          # (B, C, k, d)
        cen = _unit(e.mean(axis=2))                         # (B, C, d)
        final = cen[:, -1]                                  # (B, d)
        drift = 1.0 - np.einsum("bcd,bd->bc", cen[:, :-1], final)
        out["traj_sem"][a:b] = drift.mean(axis=1)
        e10 = e[:, -1]                                      # (B, k, d)
        pair = np.einsum("bid,bjd->b", e10, e10) - k        # sum of off-diagonal cosines
        out["list_spread"][a:b] = 1.0 - pair / (k * (k - 1))
        h = hist_mean[a:b]
        out["hist_spread"][a:b] = 1.0 - np.linalg.norm(h, axis=1)
        out["hist_to_list"][a:b] = 1.0 - np.einsum("bd,bd->b", _unit(h), final)
    return out


def compute_split(cfg, run, split, data, emb):
    from trajrisk import diffrec_data as dd
    items_df, inter, cases = data
    d = Path(cfg["paths"]["output_dir"]) / "trajlogs" / f"{run}__{split}"
    with np.load(d / "trajectory.npz") as z:
        items = z["items"]
    case_ids = pd.read_parquet(d / "cases.parquet")["case_id"].to_numpy()
    part = cases[cases["split"] == split].set_index("case_id").loc[case_ids].reset_index()
    hist, _ = dd.case_matrices(part, inter, len(items_df))
    counts = np.maximum(np.asarray(hist.sum(axis=1)).ravel(), 1)
    hist_mean = np.asarray(hist @ emb) / counts[:, None]
    sig = text_signals(items, hist_mean.astype(np.float32), emb)
    df = pd.DataFrame({"case_id": case_ids, **sig})
    if not np.isfinite(df[TEXT_ALL].to_numpy()).all():
        raise RuntimeError(f"{run} {split}: non-finite text signal")
    df.to_parquet(d / "text_signals.parquet", index=False)
    return df


def load_with_text(cfg, run, split):
    df = judge.load_split(cfg, run, split)
    d = Path(cfg["paths"]["output_dir"]) / "trajlogs" / f"{run}__{split}"
    txt = pd.read_parquet(d / "text_signals.parquet")
    m = df.merge(txt, on="case_id", how="inner", validate="one_to_one")
    if len(m) != len(df):
        raise RuntimeError(f"{run} {split}: text signals do not cover all cases")
    return m


def text_sets(sets):
    controls = sets["controls"]
    lists = [c for c in sets["controls + lists"] if c not in controls]
    traj = sets["trajectory"]
    return {
        "controls": controls,
        "controls + lists": controls + lists,
        "combined": sets["combined"],
        "controls + text": controls + TEXT_STATIC,
        "controls + text + lists": controls + TEXT_STATIC + lists,
        "controls + text + trajectory": controls + TEXT_STATIC + traj,
        "controls + lists + semantic drift": controls + lists + TEXT_TRAJ,
        "controls + text + lists + semantic drift": controls + TEXT_STATIC + lists + TEXT_TRAJ,
    }


GAINS = [  # (label, top, base)
    ("Text (static) over controls", "controls + text", "controls"),
    ("Lists over controls + text", "controls + text + lists", "controls + text"),
    ("Trajectory over controls + text", "controls + text + trajectory", "controls + text"),
    ("Semantic drift over controls + lists (proposal: generic semantic list similarity)",
     "controls + lists + semantic drift", "controls + lists"),
    ("Text (static) over combined controls + lists", "controls + text + lists", "controls + lists"),
]


def evaluate_run(cfg, run, sets):
    fit, test = load_with_text(cfg, run, judge.FIT_SPLIT), load_with_text(cfg, run, judge.EVAL_SPLIT)
    res = {"sets": {}, "single": {}}
    for name, cols in sets.items():
        prob, _ = je.fit_predict(fit, test, cols)
        res["sets"][name] = je.scores(prob, test["miss"], test["loss_ndcg"])
    for c in TEXT_ALL:                                       # untrained single signals
        res["single"][c] = je.scores(test[c].to_numpy(), test["miss"], test["loss_ndcg"])["auroc"]
    return res


def write_report(name, backbone, per_run, sets, path):
    runs = list(per_run)
    f = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
    lines = [f"# Section 7: text (MiniLM) baselines, {name} ({backbone})", "",
             f"Risk model fitted on {judge.FIT_SPLIT}, applied once to {judge.EVAL_SPLIT}; "
             "mean over runs with 95% t-interval. Controls = all final-output and history "
             "controls of the main analysis.", "", "## AUROC gains on test", "",
             "| Comparison | Gain [95% CI] |", "|---|---|"]
    gains = {}
    for label, top, base in GAINS:
        v = je.t_interval([per_run[r]["sets"][top]["auroc"] - per_run[r]["sets"][base]["auroc"]
                           for r in runs])
        gains[label] = list(v)
        lines.append(f"| {label} | {f(v)} |")
    lines += ["", "## AUROC and Brier of each predictor set (test)", "",
              "| Predictors | AUROC | Brier |", "|---|---|---|"]
    for s in sets:
        a = je.t_interval([per_run[r]["sets"][s]["auroc"] for r in runs])
        b = je.t_interval([per_run[r]["sets"][s]["brier"] for r in runs])
        lines.append(f"| `{s}` | {a[0]:.3f} | {b[0]:.4f} |")
    lines += ["", "## Untrained single text signals (test AUROC; 0.5 = chance)", "",
              "| Signal | AUROC |", "|---|---|"]
    for c in TEXT_ALL:
        lines.append(f"| `{c}` | {np.mean([per_run[r]['single'][c] for r in runs]):.3f} |")
    lines += ["", "Predictor sets: " + "; ".join(f"**{k}** = {', '.join(v)}" for k, v in sets.items())]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps(
        {"dataset": name, "backbone": backbone, "gains": gains, "runs": per_run},
        indent=2, default=float))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--skip-compute", action="store_true",
                        help="reuse existing text_signals.parquet files")
    args = parser.parse_args()
    cfg = judge.load_cfg(args.config)
    name = Path(args.config).stem
    log = Path(cfg["paths"]["output_dir"]) / "trajlogs"
    backbones = {json.loads((log / f"{r}__{judge.EVAL_SPLIT}" / "manifest.json").read_text())
                 .get("backbone", "DiffRec") for r in args.runs}
    if len(backbones) != 1:
        raise SystemExit(f"Mix of backbones {backbones}: run DiffRec and GiffCF separately.")
    backbone = backbones.pop()
    suffix = "" if backbone == "DiffRec" else "_giffcf"

    if not args.skip_compute:
        from trajrisk import diffrec_data as dd
        data = dd.load_processed(cfg)
        emb_path = Path(cfg["paths"]["data_dir"]) / "processed" / "item_emb.npy"
        emb = _unit(np.load(emb_path).astype(np.float32))
        if emb.shape[0] != len(data[0]):
            raise SystemExit(f"{emb.shape[0]} embeddings for {len(data[0])} items: rerun embed_items.py")
        for run in args.runs:
            for split in SPLITS:
                df = compute_split(cfg, run, split, data, emb)
                print(f"{run} {split}: {len(df):,} cases", flush=True)

    chosen = json.loads((judge.REPO / "reports" / "signals" / f"{name}{suffix}.json").read_text())
    base_sets, _ = judge.predictor_sets(chosen["chosen"])
    sets = text_sets(base_sets)
    per_run = {run: evaluate_run(cfg, run, sets) for run in args.runs}
    out = judge.REPO / "reports" / "text"
    out.mkdir(parents=True, exist_ok=True)
    print(write_report(name, backbone, per_run, sets, out / f"{name}{suffix}"))


if __name__ == "__main__":
    main()
