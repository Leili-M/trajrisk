"""Shortened histories for the controlled experiment (RQ3, 'why')."""

import numpy as np
import scipy.sparse as sp


def _earlier_liked(cases, inter):
    """One row per (case, earlier liked item), with the case's row number."""
    pos = inter.loc[inter["positive"], ["user_id", "item_idx", "date"]]
    pos = pos.rename(columns={"date": "date_event"})
    c = cases[["user_id", "date"]].reset_index(drop=True)
    c["row"] = np.arange(len(c))
    j = c.merge(pos, on="user_id")
    return j[j["date_event"] < j["date"]], len(c)


def _to_matrix(j, n_rows, n_items):
    data = np.ones(len(j), dtype=np.float32)
    return sp.csr_matrix((data, (j["row"].to_numpy(), j["item_idx"].to_numpy())),
                         shape=(n_rows, n_items))


def last_m_history(cases, inter, n_items, m):
    """Keep each case's m most recent earlier liked items (ties broken by item index).

    Same definition of history as diffrec_data.case_matrices. Rows follow `cases`.
    """
    j, n = _earlier_liked(cases, inter)
    j = j.sort_values(["row", "date_event", "item_idx"], ascending=[True, False, False])
    return _to_matrix(j.groupby("row", sort=False).head(m), n, n_items)


def random_m_history(cases, inter, n_items, m, seed=0):
    """Keep m earlier liked items chosen at random: same length as last_m_history,
    but without the recency of the kept items."""
    j, n = _earlier_liked(cases, inter)
    j = j.assign(key=np.random.default_rng(seed).random(len(j)))
    j = j.sort_values(["row", "key"])
    return _to_matrix(j.groupby("row", sort=False).head(m), n, n_items)