"""Turn the processed data into DiffRec inputs.

Matrices are kept sparse (only the 1s are stored) and made dense one batch
at a time, because the full dense training matrix does not fit in memory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def load_processed(cfg):
    folder = Path(cfg["paths"]["data_dir"]) / "processed"
    items = pd.read_parquet(folder / "items.parquet")
    inter = pd.read_parquet(folder / "interactions.parquet")
    cases = pd.read_parquet(folder / "cases.parquet")
    return items, inter, cases


def _to_csr(rows, cols, n_rows, n_items):
    mat = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                        shape=(n_rows, n_items))
    mat.data[:] = 1.0          # a (row, item) pair appearing twice still counts as 1
    return mat


def train_matrix(inter, n_items):
    """One row per training user: 1 for each recipe liked in the training period."""
    train = inter[(inter["split"] == "train") & inter["positive"]]
    users = np.sort(train["user_id"].unique())
    rows = np.searchsorted(users, train["user_id"].to_numpy())
    return _to_csr(rows, train["item_idx"].to_numpy(), len(users), n_items), users


def case_matrices(cases, inter, n_items):
    """Two matrices with one row per case, in the order of `cases`.

    history: recipes the user liked on strictly earlier days (model input)
    seen:    recipes the user rated, with any rating, on strictly earlier days
             (removed from the recommendations)
    """
    c = cases[["user_id", "date"]].reset_index(drop=True)
    c["row"] = np.arange(len(c))

    def earlier(events):
        m = c.merge(events[["user_id", "item_idx", "date"]],
                    on="user_id", suffixes=("", "_event"))
        m = m[m["date_event"] < m["date"]]
        return _to_csr(m["row"].to_numpy(), m["item_idx"].to_numpy(), len(c), n_items)

    history = earlier(inter[inter["positive"]])
    seen = earlier(inter)
    return history, seen


def iterate_batches(mat, batch_size, shuffle=False, seed=0):
    """Yield (row indices, dense float tensor) one batch at a time."""
    order = np.arange(mat.shape[0])
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        yield idx, torch.from_numpy(mat[idx].toarray())
