"""Evaluation of the controlled track (proposal sections 6-8 for the benchmark).

Input: one row per (run, scenario, pipeline) with signals (preference.scenario_signals)
and outcomes (preference.scenario_outcome). Pipelines: plain (no request), soft
(request-aware reranker, violations possible), hard (only feasible recipes).
All thresholds and the judge are fitted on validation scenarios; test scenarios are
used once. Scenario families (all runs of a scenario) are resampled together.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .preference import FINAL_CHECKS, SIGNALS_IDENTITY, SIGNALS_OTHER, SIGNALS_PREFERENCE

# orientation: every detector score must be "higher = riskier"
ORIENT = {"util_final": -1.0}
DETECTORS = SIGNALS_IDENTITY + SIGNALS_PREFERENCE + SIGNALS_OTHER + FINAL_CHECKS
MATCHED_RECALL = 0.8


def risk(df, col):
    return ORIENT.get(col, 1.0) * df[col].to_numpy(dtype=float)


def auc(y, s):
    y = np.asarray(y, dtype=bool)
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def detection_rows(df, pipeline):
    """Rows usable for failure detection: scenarios with feasible recipes."""
    d = df[(df["pipeline"] == pipeline) & (df["feasible_type"] != "none") & df["failure"].notna()]
    return d[d["split"] == "valid"], d[d["split"] == "test"]


def outcome_table(df):
    rows = []
    t = df[df["split"] == "test"]
    for p, g in t.groupby("pipeline"):
        feas = g[g["feasible_type"] != "none"]
        none = g[g["feasible_type"] == "none"]
        rows.append({
            "pipeline": p,
            "mean regret (many)": feas.loc[feas["feasible_type"] == "many", "regret"].mean(),
            "mean regret (few)": feas.loc[feas["feasible_type"] == "few", "regret"].mean(),
            "failure rate": feas["failure"].astype(float).mean(),
            "lists with a violation": (feas["n_violating"] > 0).mean(),
            "abstained when nothing feasible": (none["list_len"] == 0).mean() if len(none) else np.nan,
            "regret, held-out combinations": feas.loc[feas["heldout"], "regret"].mean(),
            "regret, other": feas.loc[~feas["heldout"], "regret"].mean(),
            "diversity": feas["diversity"].mean(),
        })
    return pd.DataFrame(rows)


def matched_threshold(valid, col, recall=MATCHED_RECALL):
    """Threshold that flags `recall` of the validation failures (flag if risk >= t)."""
    s = risk(valid, col)[valid["failure"].astype(bool).to_numpy()]
    return float(np.quantile(s, 1 - recall)) if len(s) else np.nan


def single_signal_table(df, pipeline):
    valid, test = detection_rows(df, pipeline)
    y = test["failure"].astype(bool).to_numpy()
    rows = []
    for col in DETECTORS:
        s = risk(test, col)
        if np.nanstd(s) == 0:
            continue
        t = matched_threshold(valid, col)
        flag = s >= t
        rows.append({"signal": col, "auroc": auc(y, s),
                     "auroc_mean_over_runs": np.nanmean([auc(g["failure"].astype(bool), risk(g, col))
                                                         for _, g in test.groupby("run")]),
                     "recall at matched threshold": flag[y].mean() if y.any() else np.nan,
                     "false alarms among suitable": flag[~y].mean() if (~y).any() else np.nan})
    return pd.DataFrame(rows), int(y.sum()), int(len(y))


def _fit(train, cols):
    X = np.column_stack([risk(train, c) for c in cols])
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(
        X, train["failure"].astype(bool))


def _pred(clf, d, cols):
    return clf.predict_proba(np.column_stack([risk(d, c) for c in cols]))[:, 1]


def judge_sets():
    base = FINAL_CHECKS + ["log_history"]
    return {"final checks": base,
            "final checks + identity": base + SIGNALS_IDENTITY,
            "final checks + preference": base + SIGNALS_PREFERENCE,
            "final checks + identity + preference": base + SIGNALS_IDENTITY + SIGNALS_PREFERENCE}


def judge_table(df, pipeline, b=500, seed=0):
    """Judge fitted on validation scenarios, AUROC on test, paired scenario bootstrap."""
    valid, test = detection_rows(df, pipeline)
    if valid["failure"].nunique() < 2 or test["failure"].nunique() < 2:
        return pd.DataFrame(), {}
    sets = judge_sets()
    usable = {k: [c for c in v if np.nanstd(risk(valid, c)) > 0] for k, v in sets.items()}
    preds = {k: _pred(_fit(valid, cols), test, cols) for k, cols in usable.items()}
    y = test["failure"].astype(bool).to_numpy()
    rows = [{"predictors": k, "auroc": auc(y, p)} for k, p in preds.items()]
    pairs = [("final checks", "final checks + identity"),
             ("final checks", "final checks + preference"),
             ("final checks + identity", "final checks + identity + preference")]
    rng = np.random.default_rng(seed)
    codes, n = pd.factorize(test["scenario_id"])[0], test["scenario_id"].nunique()
    boot = {p: [] for p in pairs}
    for _ in range(b):
        wts = rng.multinomial(n, np.full(n, 1 / n))[codes].astype(float)
        keep = wts > 0
        if len(np.unique(y[keep])) < 2:
            continue
        for base, top in pairs:
            boot[(base, top)].append(
                roc_auc_score(y[keep], preds[top][keep], sample_weight=wts[keep])
                - roc_auc_score(y[keep], preds[base][keep], sample_weight=wts[keep]))
    gains = {f"{t} over {bse}": (auc(y, preds[t]) - auc(y, preds[bse]),
                                 float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
             for (bse, t), v in boot.items() if v}
    return pd.DataFrame(rows), gains


def subset_table(df, pipeline, signals=("u_set", "u_rank", "u_attr", "u_util")):
    """Proposal diagnostic subsets on test, with 'changing' fixed by the validation median
    of membership change (identity-based instability):
      changing-but-suitable, consequentially unstable, stable-but-unsuitable, stable-and-suitable.
    For each subset: size, mean regret, and the share flagged by each signal at its
    matched-recall threshold (flags among suitable lists are false alarms)."""
    valid, test = detection_rows(df, pipeline)
    med = float(np.median(valid["u_set"]))
    changing = test["u_set"].to_numpy() > med
    fail = test["failure"].astype(bool).to_numpy()
    names = {(True, False): "changing but suitable", (True, True): "consequentially unstable",
             (False, True): "stable but unsuitable", (False, False): "stable and suitable"}
    thr = {c: matched_threshold(valid, c) for c in signals}
    rows = []
    for (ch, fl), name in names.items():
        m = (changing == ch) & (fail == fl)
        row = {"subset": name, "cases": int(m.sum()),
               "mean regret": float(test.loc[m, "regret"].mean()) if m.any() else np.nan}
        for c in signals:
            row[f"flagged by {c}"] = float((risk(test, c)[m] >= thr[c]).mean()) if m.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows), med
