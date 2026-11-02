"""Proposal section 6 (RQ2): preference-aware trajectory comparison on the controlled
recipe benchmark (600 HUMMUS scenarios from section 3).

For every DiffRec HUMMUS run and every scenario, the trajectory of the scenario's real
user history is computed. An identical request-aware reranker (proposal formula 9) is
applied at every checkpoint:
    score = z(model score) + beta * soft utility - gamma * known violations
Pipelines: plain (no request, beta = gamma = 0), soft (violations possible),
hard (only known-feasible recipes; an empty list = abstention).
beta and gamma are chosen on the 200 validation scenarios (lowest mean regret of the soft
pipeline, final lists pooled over runs); test scenarios are used once.

Outputs: <output_dir>/preference/rows.parquet (one row per run x scenario x pipeline),
lists.npz; reports/preference/hummus.md and .json.

    sbatch --partition=gpumedium --time=02:00:00 jobs/run.sh configs/base.yaml \
        scripts/preference_experiment.py --config configs/base.yaml \
        --runs hm_seed42 hm_seed1 hm_seed2 hm_seed3 hm_seed4
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajrisk import benchmark as bm
from trajrisk import diffrec_data as dd
from trajrisk import preference as pf
from trajrisk import preference_eval as pe
from trajrisk.trajectory import diffrec_trajectory

REPO = Path(__file__).resolve().parents[1]
BETAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
GAMMAS = (0.0, 1.0, 10.0)


def md(df):
    """Markdown table without extra packages."""
    cols = list(df.columns)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


def load_diffrec(cfg, run, n_items, device):
    run_dir = Path(cfg["paths"]["output_dir"]) / "models" / run
    meta = json.loads((run_dir / "config.json").read_text())
    from trajrisk.diffrec_model import build_diffrec
    model, diffusion = build_diffrec(dict(cfg, model=meta["model"]), n_items, device)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model, diffusion


def scenario_inputs(scen, attrs, pref):
    """Per scenario: utility, feasibility, violations, unknown status, weight vector."""
    out = []
    for s in scen:
        hard = {k: v for k, v in s["hard"].items() if v is not None}
        status = bm.hard_status(attrs, hard)
        feasible = (status == 1).all(axis=1)
        viol = (status == 0).sum(axis=1)
        out.append({"util": bm.item_utility(pref, s["soft"]), "feasible": feasible, "viol": viol,
                    "unknown": ~feasible & (viol == 0),
                    "w": np.array([s["soft"].get(n, 0.0) for n in pf.PREF_NAMES], dtype=float)})
    return out


def lists_for(traj_z, seen, inp, beta, gamma, hard):
    """(C, K) lists: the same reranker at every checkpoint."""
    return np.stack([pf.rerank(traj_z[c], inp["util"], inp["viol"], beta, gamma, seen,
                               inp["feasible"] if hard else None) for c in range(len(traj_z))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    store = Path(cfg["paths"]["data_dir"]) / "processed" / "benchmark"
    attrs = pd.read_pickle(store / "item_attributes.pkl").sort_values("item_idx").reset_index(drop=True)
    pref = bm.preference_scores(attrs)
    P = pref[pf.PREF_NAMES].to_numpy(dtype=float)
    scen = [json.loads(line) for line in open(store / "scenarios.jsonl")]
    items, inter, cases = dd.load_processed(cfg)
    n_items = len(items)
    if not (attrs["item_idx"].to_numpy() == np.arange(n_items)).all():
        raise RuntimeError("item_attributes.pkl is not aligned with the catalog")
    part = cases.set_index("case_id").loc[[s["case_id"] for s in scen]].reset_index()
    hist, seen_m = dd.case_matrices(part, inter, n_items)
    seen = [seen_m[i].toarray().ravel() > 0 for i in range(len(scen))]
    inputs = scenario_inputs(scen, attrs, pref)
    print(f"{len(scen)} scenarios, {n_items} recipes, inputs ready in {time.time() - t0:.0f} s",
          flush=True)

    # 1) trajectories (normalised per checkpoint) for every run
    traj = {}
    for run in args.runs:
        model, diffusion = load_diffrec(cfg, run, n_items, device)
        chunks = []
        with torch.no_grad():
            for idx, h in dd.iterate_batches(hist, args.batch_size):
                chunks.append(diffrec_trajectory(model, diffusion, h.to(device)).float().cpu().numpy())
        t = np.concatenate(chunks, axis=1)                          # (C, scenarios, items)
        traj[run] = np.stack([[pf.zscore(t[c, i], seen[i]) for c in range(t.shape[0])]
                              for i in range(len(scen))]).astype(np.float32)  # (scen, C, items)
        print(f"{run}: trajectory {t.shape}, {time.time() - t0:.0f} s", flush=True)

    # 2) choose beta, gamma on validation scenarios (soft pipeline, final lists)
    val_idx = [i for i, s in enumerate(scen) if s["split"] != "test" and not s["abstain"]]
    grid = []
    for beta in BETAS:
        for gamma in GAMMAS:
            regs = []
            for run in args.runs:
                for i in val_idx:
                    inp = inputs[i]
                    final = pf.rerank(traj[run][i, -1], inp["util"], inp["viol"], beta, gamma, seen[i])
                    regs.append(bm.regret(final[final >= 0], inp["util"], inp["feasible"]))
            grid.append({"beta": beta, "gamma": gamma, "valid mean regret": float(np.nanmean(regs))})
    grid = pd.DataFrame(grid)
    best = grid.loc[grid["valid mean regret"].idxmin()]
    beta, gamma = float(best["beta"]), float(best["gamma"])
    print(f"chosen beta {beta}, gamma {gamma} (valid regret {best['valid mean regret']:.3f})", flush=True)

    # 3) lists, signals and outcomes
    rows, saved = [], {}
    pipes = {"plain": (0.0, 0.0, False), "soft": (beta, gamma, False), "hard": (beta, gamma, True)}
    for run in args.runs:
        for i, s in enumerate(scen):
            inp = inputs[i]
            for name, (b, g, hard) in pipes.items():
                L = lists_for(traj[run][i], seen[i], inp, b, g, hard)
                saved[f"{run}|{s['scenario_id']}|{name}"] = L
                row = {"run": run, "scenario_id": s["scenario_id"],
                       "split": "test" if s["split"] == "test" else "valid", "pipeline": name,
                       "feasible_type": s["feasible_type"], "soft_type": s["soft_type"],
                       "heldout": bool(s["heldout_soft"] or s["heldout_course"]),
                       "log_history": float(np.log1p(part.loc[i, "n_prior"])),
                       "diversity": pf.intra_list_diversity(P, L[-1])}
                row.update(pf.scenario_signals(L, P, inp["w"], inp["util"], inp["viol"],
                                               inp["unknown"]))
                row.update(pf.scenario_outcome(L[-1], inp["util"], inp["feasible"], inp["viol"]))
                rows.append(row)
        print(f"{run}: signals done, {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    out = Path(cfg["paths"]["output_dir"]) / "preference"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "rows.parquet", index=False)
    np.savez_compressed(out / "lists.npz", keys=np.array(list(saved)), lists=np.stack(list(saved.values())))

    # 4) report
    lines = ["# Section 6: preference-aware comparison (HUMMUS benchmark, DiffRec)", "",
             f"Runs: {', '.join(args.runs)}. Reranker (formula 9) chosen on validation scenarios: "
             f"beta = {beta:g}, gamma = {gamma:g}. Failure = regret >= {pf.FAIL_REGRET}. "
             "Detection rows: scenarios with feasible recipes; thresholds and judge from validation.",
             "", "## Reranker grid (validation, soft pipeline)", "", md(grid),
             "", "## Outcomes on test", "", md(pe.outcome_table(df))]
    js = {"beta": beta, "gamma": gamma, "grid": grid.to_dict("records")}
    for pipe in ("soft", "hard"):
        single, n_fail, n = pe.single_signal_table(df, pipe)
        judge, gains = pe.judge_table(df, pipe)
        subsets, med = pe.subset_table(df, pipe)
        lines += ["", f"## Pipeline: {pipe}", "",
                  f"Test detection rows: {n} ({n_fail} failures).", "",
                  "### Single signals (test)", "", md(single), ""]
        if len(judge):
            lines += ["### Judge fitted on validation scenarios (test AUROC)", "",
                      md(judge), "",
                      "Paired scenario bootstrap (95% interval of the AUROC gain):", ""]
            lines += [f"- {k}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in gains.items()]
        lines += ["", f"### Diagnostic subsets (changing = membership change > {med:.3f}, "
                  "the validation median)", "", md(subsets)]
        js[pipe] = {"single": single.to_dict("records"), "judge": judge.to_dict("records"),
                    "gains": gains, "subsets": subsets.to_dict("records")}
    rep = REPO / "reports" / "preference"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "hummus.md").write_text("\n".join(lines) + "\n")
    (rep / "hummus.json").write_text(json.dumps(js, indent=2, default=float))
    print("\n".join(lines))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
