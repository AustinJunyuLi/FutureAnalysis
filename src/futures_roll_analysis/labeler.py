from __future__ import annotations

"""
Deterministic expiry-based strip labeler.

Provides compute_strip_labels which produces F1..Fn purely from expiry
timestamps at the given index instants. This logic is independent of any
price availability and is therefore deterministic given (index, expiries).
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def compute_strip_labels(
    ts_index_utc: pd.DatetimeIndex,
    contract_order: List[str],
    expiries_utc: Dict[str, pd.Timestamp],
    *,
    depth: int = 12,
) -> pd.DataFrame:
    """
    Compute F1..F{depth} labels using expiry timestamps only.

    Parameters
    ----------
    ts_index_utc:
        UTC tz-aware DatetimeIndex for which labels are computed.
    contract_order:
        Contracts sorted by ascending expiry (earliest first).
    expiries_utc:
        Mapping contract -> UTC expiry timestamp.
    depth:
        Number of forward labels to produce (default 12).

    Returns
    -------
    DataFrame with columns ['F1', 'F2', ...], index=ts_index_utc, dtype=object.
    For any t, Fk(t) is the k-th soonest-to-expire contract among expiries > t.
    """
    if ts_index_utc.tz is None:
        raise ValueError("ts_index_utc must be timezone-aware UTC")

    # Prepare expiry vector in UTC nanoseconds
    exp_values = []
    for c in contract_order:
        ts = expiries_utc.get(c)
        if ts is None or ts.tz is None:
            raise ValueError(f"Missing or tz-naive UTC expiry for contract {c}")
        exp_values.append(ts.tz_convert("UTC").value)
    exp = np.array(exp_values, dtype=np.int64)

    # Timestamp vector in UTC nanoseconds
    tvals = ts_index_utc.view("int64")

    # For each t, find the first index where expiry > t
    # side='right' ensures switch happens exactly at expiry instant
    i_vec = np.searchsorted(exp, tvals, side="right")

    # Build 2-D gather indices for depth labels
    steps = np.arange(depth, dtype=np.int64)[None, :]
    start = i_vec[:, None] + steps  # shape (T, depth)

    # Mask for valid indices
    valid = (start >= 0) & (start < len(contract_order))
    idx = np.where(valid, start, -1)

    # Gather labels
    base = np.array(contract_order, dtype=object)
    labels = np.empty_like(idx, dtype=object)
    flat_mask = valid.ravel()
    labels.ravel()[flat_mask] = base[idx.ravel()[flat_mask]]
    labels.ravel()[~flat_mask] = None

    # Construct DataFrame
    cols = [f"F{i}" for i in range(1, depth + 1)]
    out = pd.DataFrame(labels, index=ts_index_utc, columns=cols)
    return out

