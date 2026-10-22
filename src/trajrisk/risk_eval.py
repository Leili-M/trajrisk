"""Does an uncertainty signal predict wrong recommendations beyond history length?

The same three checks as Part 10 of the progress report:
1. AUROC of each signal on its own, and its Spearman correlation with history length.
2. AUROC inside groups of cases with similar history length (4 quartiles).
3. Cross-validated AUROC of history length alone vs. history length plus the signals,
   with folds split by user so one user never sits in both the training and the test fold.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def usable(df, cols):
    """Drop signals that are constant (e.g. sample disagreement with one noise-free sample)."""
    return [c for c in cols if df[c].nunique() > 1]


def single_signal_table(df, cols, miss="miss", history="history_size"):
    rows = []
    for c in cols:
        rows.append({"signal": c,
                     "auroc": roc_auc_score(df[miss], df[c]),
                     "spearman_with_history": spearmanr(df[c], df[history]).correlation})
    rows.append({"signal": history, "auroc": roc_auc_score(df[miss], df[history]),
                 "spearman_with_history": 1.0})
    return pd.DataFrame(rows)


def within_group_table(df, cols, miss="miss", history="history_size", n_groups=4):
    labels = [f"Q{i + 1}" for i in range(n_groups)]
    try:                                   # same grouping as Part 10
        groups = pd.qcut(df[history], n_groups, labels=labels)
    except ValueError:                     # too many ties: split ties by row order
        groups = pd.qcut(df[history].rank(method="first"), n_groups, labels=labels)
    rows = []
    for g, part in df.groupby(groups, observed=True):
        row = {"group": g, "cases": len(part), "wrong": part[miss].mean(),
               "history_min": part[history].min(), "history_max": part[history].max()}
        for c in cols:
            ok = part[miss].nunique() == 2
            row[c] = roc_auc_score(part[miss], part[c]) if ok else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def cv_auroc(df, cols, users, miss="miss", n_folds=5):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    p = cross_val_predict(clf, df[cols], df[miss], cv=GroupKFold(n_folds),
                          groups=users, method="predict_proba")[:, 1]
    return roc_auc_score(df[miss], p)


def added_value_table(df, signal_sets, users, miss="miss", history="history_size"):
    """CV AUROC of history alone and of history plus each named set of signals."""
    d = df.copy()
    d["log_history"] = np.log1p(d[history])
    base = cv_auroc(d, ["log_history"], users, miss)
    rows = [{"predictors": "history only", "cv_auroc": base, "gain": 0.0}]
    for name, cols in signal_sets.items():
        cols = usable(d, cols)
        if not cols:
            continue
        a = cv_auroc(d, ["log_history"] + cols, users, miss)
        rows.append({"predictors": f"history + {name}", "cv_auroc": a, "gain": a - base})
    return pd.DataFrame(rows)


def evaluate(df, signal_sets, users, miss="miss", history="history_size"):
    """All three checks. signal_sets maps a name to a list of signal columns."""
    df = df.copy()
    df[miss] = df[miss].astype(bool)
    all_cols = usable(df, sorted({c for cols in signal_sets.values() for c in cols}))
    return {
        "single": single_signal_table(df, all_cols, miss, history),
        "within_groups": within_group_table(df, all_cols, miss, history),
        "added_value": added_value_table(df, signal_sets, users, miss, history),
    }





def cv_predict(df, cols, users, miss="miss", n_folds=5):

    """Out-of-fold probability of a wrong recommendation (folds split by user)."""

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    return cross_val_predict(clf, df[cols], df[miss], cv=GroupKFold(n_folds),

                             groups=users, method="predict_proba")[:, 1]

