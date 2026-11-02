"""Risk model ("judge") and its evaluation (proposal sections 7 and 8).

The judge is a logistic regression fitted on the risk-validation cases and applied
once to the test cases (proposal: thresholds and calibration only from validation).
It is supervised reliability estimation: it is fitted to failure labels.

Metrics (section 8):
  AUROC, AUPRC (with the failure prevalence), Brier score,
  risk-coverage area: keep the cases with the lowest predicted risk, compute the mean
  loss of the kept cases at every coverage, and average over coverages (lower = better),
  with loss = miss or loss = 1 - NDCG@K,
  reliability table: predicted vs observed failure rate in equal-size bins.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .risk_eval import check_predictors

K = 10


def labels_from_ranks(rank_final, k=K):
    """miss = target not in top-k; loss_ndcg = 1 - NDCG@k for one relevant item (0-based rank)."""
    rank = np.asarray(rank_final)
    miss = rank >= k
    ndcg = np.where(miss, 0.0, 1.0 / np.log2(rank + 2.0))
    return miss, 1.0 - ndcg


def make_judge():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def fit_predict(train, test, cols, label="miss"):
    """Fit on train (risk-validation), predict failure probability on test."""
    check_predictors(cols)
    clf = make_judge().fit(train[cols], train[label])
    return clf.predict_proba(test[cols])[:, 1], clf


def cv_predict(df, cols, groups, label="miss", n_folds=5):
    """Out-of-fold predictions inside one split, folds split by user (earlier protocol)."""
    check_predictors(cols)
    return cross_val_predict(make_judge(), df[cols], df[label], cv=GroupKFold(n_folds),
                             groups=groups, method="predict_proba")[:, 1]


def risk_coverage(risk, loss):
    """Mean loss of the kept cases at every coverage (cases kept by increasing risk)."""
    order = np.argsort(risk, kind="stable")
    kept_loss = np.cumsum(np.asarray(loss, dtype=float)[order])
    n = np.arange(1, len(order) + 1)
    return n / len(order), kept_loss / n


def aurc(risk, loss):
    """Area under the risk-coverage curve (mean over all coverages); lower is better."""
    return float(risk_coverage(risk, loss)[1].mean())


def reference_aurc(loss, seed=0):
    """Diagnostic references: random order and outcome-based oracle order."""
    loss = np.asarray(loss, dtype=float)
    rng = np.random.default_rng(seed)
    return {"random": aurc(rng.random(len(loss)), loss), "oracle": aurc(loss, loss)}


def reliability_table(prob, miss, n_bins=10):
    """Equal-size bins of predicted probability: mean predicted vs observed failure rate."""
    d = pd.DataFrame({"p": prob, "y": np.asarray(miss, dtype=float)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    t = d.groupby("bin").agg(cases=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return t.reset_index()


def scores(prob, miss, loss_ndcg):
    miss = np.asarray(miss, dtype=bool)
    return {
        "auroc": float(roc_auc_score(miss, prob)),
        "auprc": float(average_precision_score(miss, prob)),
        "prevalence": float(miss.mean()),
        "brier": float(brier_score_loss(miss, np.clip(prob, 0, 1))),
        "aurc_miss": aurc(prob, miss),
        "aurc_ndcg": aurc(prob, loss_ndcg),
    }


def t_interval(values, level=0.95):
    """Mean and t-interval over seeds."""
    from scipy.stats import t
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return m, np.nan, np.nan
    h = float(t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return m, m - h, m + h
