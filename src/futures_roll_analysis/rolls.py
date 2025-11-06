from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .labeler import compute_strip_labels

def build_expiry_map(expiry_df: pd.DataFrame, *, default_hour: int = 17, default_minute: int = 0) -> pd.Series:
    """
    Return a Series mapping contract codes to naive expiry timestamps (local day + time).

    Notes
    -----
    - Historically we normalized to midnight (date-only). For better alignment with
      actual switch moments, we now attach a default local time-of-day (naive) to
      each expiry date. This avoids day-boundary ambiguities while keeping timezone
      handling unchanged elsewhere (panel indices are naive local times).
    - If callers need timezone-aware timestamps, they should localize explicitly.
    """
    if "contract" not in expiry_df.columns or "expiry_date" not in expiry_df.columns:
        raise ValueError("Expiry metadata must include 'contract' and 'expiry_date' columns")

    df = expiry_df.drop_duplicates("contract").copy()
    df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.normalize()
    # Attach default expiry time within the local day (naive)
    df["expiry_ts"] = df["expiry_date"] + pd.to_timedelta(default_hour, unit="h") + pd.to_timedelta(default_minute, unit="m")
    return df.set_index("contract")["expiry_ts"]


def identify_front_next(
    panel: pd.DataFrame,
    expiry_map: pd.Series,
    *,
    price_field: str = "close",
    meta_namespace: str = "meta",
) -> pd.DataFrame:
    """
    Vectorised front/next identification for each row in a panel.

    Parameters
    ----------
    panel:
        DataFrame with MultiIndex columns ``(contract, field)``.
    expiry_map:
        Series mapping contracts to expiry ``Timestamp`` objects.
    price_field:
        Column to use for evaluating whether a contract is active (default ``"close"``).
    meta_namespace:
        Name of the metadata column namespace inside the panel.
    """
    contracts = [
        contract
        for contract in panel.columns.get_level_values(0).unique()
        if contract != meta_namespace
    ]

    close_df = _extract_field(panel, contracts, price_field)
    if close_df.empty:
        raise ValueError("Panel is missing close field data")

    expiry_series = expiry_map.reindex(contracts)
    if expiry_series.isna().any():
        missing = expiry_series[expiry_series.isna()].index.tolist()
        raise ValueError(f"Expiry metadata missing for contracts: {missing}")

    dates = pd.to_datetime(
        panel.index.get_level_values(0) if isinstance(panel.index, pd.MultiIndex) else panel.index
    )

    close_values = close_df.to_numpy(dtype=float)

    expiry_array = pd.to_datetime(expiry_series).to_numpy(dtype="datetime64[ns]")
    expiry_int = expiry_array.astype("datetime64[ns]").astype("int64")
    date_int = dates.to_numpy(dtype="datetime64[ns]").astype("int64")

    delta = expiry_int.reshape(1, -1) - date_int.reshape(-1, 1)
    # Define activeness purely by time (decouple from data availability)
    active_mask = (delta >= 0)

    delta = delta.astype("float64")
    delta[~active_mask] = np.inf

    front_idx = delta.argmin(axis=1)
    front_delta = delta[np.arange(len(delta)), front_idx]
    front_valid = np.isfinite(front_delta)

    delta_next = delta.copy()
    delta_next[np.arange(len(delta_next)), front_idx] = np.inf
    next_idx = delta_next.argmin(axis=1)
    next_delta = delta_next[np.arange(len(delta_next)), next_idx]
    next_valid = np.isfinite(next_delta)

    contract_array = np.asarray(contracts, dtype="object")
    front_contracts = np.where(front_valid, contract_array[front_idx], None)
    next_contracts = np.where(next_valid, contract_array[next_idx], None)

    result = pd.DataFrame(
        {
            "front_contract": front_contracts,
            "next_contract": next_contracts,
        },
        index=panel.index,
    )
    return result


def identify_front_to_f12(
    panel: pd.DataFrame,
    expiry_map: pd.Series,
    *,
    max_contracts: int = 12,
    price_field: str = "close",
    meta_namespace: str = "meta",
) -> pd.DataFrame:
    """
    Vectorised identification of F1, F2, ..., F_max_contracts for each panel row.

    Extends identify_front_next() to find the nearest `max_contracts` unexpired contracts
    at each timestamp.

    Parameters
    ----------
    panel:
        DataFrame with MultiIndex columns ``(contract, field)``.
    expiry_map:
        Series mapping contracts to expiry ``Timestamp`` objects.
    max_contracts:
        Number of contract levels to identify (default 12 for F1...F12).
    price_field:
        Column to use for evaluating whether a contract is active (default ``"close"``).
    meta_namespace:
        Name of the metadata column namespace inside the panel.

    Returns
    -------
    DataFrame with columns ['F1', 'F2', ..., 'F{max_contracts}']
    """
    contracts = [
        contract
        for contract in panel.columns.get_level_values(0).unique()
        if contract != meta_namespace
    ]

    close_df = _extract_field(panel, contracts, price_field)
    if close_df.empty:
        raise ValueError("Panel is missing close field data")

    expiry_series = expiry_map.reindex(contracts)
    if expiry_series.isna().any():
        missing = expiry_series[expiry_series.isna()].index.tolist()
        raise ValueError(f"Expiry metadata missing for contracts: {missing}")

    dates = pd.to_datetime(
        panel.index.get_level_values(0) if isinstance(panel.index, pd.MultiIndex) else panel.index
    )

    close_values = close_df.to_numpy(dtype=float)

    expiry_array = pd.to_datetime(expiry_series).to_numpy(dtype="datetime64[ns]")
    expiry_int = expiry_array.astype("datetime64[ns]").astype("int64")
    date_int = dates.to_numpy(dtype="datetime64[ns]").astype("int64")

    delta = expiry_int.reshape(1, -1) - date_int.reshape(-1, 1)
    # Define activeness purely by time (decouple from data availability)
    active_mask = (delta >= 0)

    delta = delta.astype("float64")
    delta[~active_mask] = np.inf

    contract_array = np.asarray(contracts, dtype="object")
    result_dict = {}

    # Iteratively find F1, F2, ..., F_max_contracts
    delta_working = delta.copy()
    for i in range(1, max_contracts + 1):
        contract_idx = delta_working.argmin(axis=1)
        contract_delta = delta_working[np.arange(len(delta_working)), contract_idx]
        contract_valid = np.isfinite(contract_delta)

        contract_names = np.where(contract_valid, contract_array[contract_idx], None)
        result_dict[f"F{i}"] = contract_names

        # Mark selected contracts as unavailable for next iteration
        delta_working[np.arange(len(delta_working)), contract_idx] = np.inf

    result = pd.DataFrame(result_dict, index=panel.index)
    return result


def identify_front_to_f12_v2(
    panel: pd.DataFrame,
    expiry_map: pd.Series,
    *,
    tz_exchange: str = "America/Chicago",
    max_contracts: int = 12,
    meta_namespace: str = "meta",
) -> pd.DataFrame:
    """
    Deterministic identification of F1..F{max_contracts} using expiry timestamps only.

    Converts the panel index to UTC (assuming exchange-local tz for naive indices)
    and expiries to UTC, then uses compute_strip_labels to build the strip.
    """
    ts = panel.index.get_level_values(0) if isinstance(panel.index, pd.MultiIndex) else panel.index
    idx = pd.DatetimeIndex(ts)
    if idx.tz is None:
        idx_utc = idx.tz_localize(tz_exchange, nonexistent="shift_forward", ambiguous="NaT").tz_convert("UTC")
    else:
        idx_utc = idx.tz_convert("UTC")

    # Contracts present in the panel
    contracts = [c for c in panel.columns.get_level_values(0).unique() if c != meta_namespace]
    # Sort by expiry ascending
    exp_series = pd.to_datetime(expiry_map.reindex(contracts))
    contracts_sorted = exp_series.sort_values().index.tolist()

    # Convert expiries to UTC tz-aware
    expiries_utc = {}
    for c in contracts_sorted:
        v = exp_series.loc[c]
        if pd.isna(v):
            continue
        ts_local = pd.Timestamp(v)
        if ts_local.tz is None:
            ts_local = ts_local.tz_localize(tz_exchange, ambiguous="NaT", nonexistent="shift_forward")
        expiries_utc[c] = ts_local.tz_convert("UTC")

    strip = compute_strip_labels(idx_utc, contracts_sorted, expiries_utc, depth=max_contracts)
    strip.index = panel.index
    return strip


def identify_front_next_v2(
    panel: pd.DataFrame,
    expiry_map: pd.Series,
    *,
    tz_exchange: str = "America/Chicago",
    meta_namespace: str = "meta",
) -> pd.DataFrame:
    """Front/next (F1/F2) wrapper over the deterministic labeler."""
    strip = identify_front_to_f12_v2(
        panel,
        expiry_map,
        tz_exchange=tz_exchange,
        max_contracts=2,
        meta_namespace=meta_namespace,
    ).rename(columns={"F1": "front_contract", "F2": "next_contract"})
    return strip


def compute_spread(
    panel: pd.DataFrame,
    front_next: pd.DataFrame,
    *,
    price_field: str = "close",
    meta_namespace: str = "meta",
) -> pd.Series:
    """Compute calendar spreads (next minus front) using vectorised indexing."""
    contracts = [
        contract
        for contract in panel.columns.get_level_values(0).unique()
        if contract != meta_namespace
    ]
    close_df = _extract_field(panel, contracts, price_field)
    values = close_df.to_numpy(dtype=float)

    front_contracts = front_next["front_contract"].to_numpy(dtype=object)
    next_contracts = front_next["next_contract"].to_numpy(dtype=object)

    contract_index = {contract: idx for idx, contract in enumerate(contracts)}
    front_indices = np.array([contract_index.get(c, -1) for c in front_contracts])
    next_indices = np.array([contract_index.get(c, -1) for c in next_contracts])

    front_prices = np.full(len(values), np.nan)
    next_prices = np.full(len(values), np.nan)

    valid_front = front_indices >= 0
    valid_next = next_indices >= 0

    if valid_front.any():
        row_idx = np.nonzero(valid_front)[0]
        front_prices[valid_front] = values[row_idx, front_indices[valid_front]]
    if valid_next.any():
        row_idx = np.nonzero(valid_next)[0]
        next_prices[valid_next] = values[row_idx, next_indices[valid_next]]

    spread = pd.Series(next_prices - front_prices, index=panel.index, name=f"spread_{price_field}")
    return spread


def compute_multi_spreads(
    panel: pd.DataFrame,
    contract_chain: pd.DataFrame,
    *,
    price_field: str = "close",
    meta_namespace: str = "meta",
) -> pd.DataFrame:
    """
    Compute S1, S2, ..., S_n spreads from contract chain.

    Parameters
    ----------
    panel:
        DataFrame with MultiIndex columns ``(contract, field)``.
    contract_chain:
        DataFrame with columns ['F1', 'F2', ..., 'F{n}'] from identify_front_to_f12().
    price_field:
        Price field to use (default ``"close"``).
    meta_namespace:
        Name of the metadata column namespace inside the panel.

    Returns
    -------
    DataFrame with columns ['S1', 'S2', ..., 'S{n-1}'] where S_i = F_{i+1} - F_i
    """
    contracts = [
        contract
        for contract in panel.columns.get_level_values(0).unique()
        if contract != meta_namespace
    ]
    close_df = _extract_field(panel, contracts, price_field)
    values = close_df.to_numpy(dtype=float)

    contract_index = {contract: idx for idx, contract in enumerate(contracts)}

    # Determine how many contract levels we have
    f_columns = [col for col in contract_chain.columns if col.startswith('F')]
    num_contracts = len(f_columns)

    result_dict = {}

    # Compute S_i = F_{i+1} - F_i for i = 1 to num_contracts-1
    for i in range(1, num_contracts):
        f_i_col = f"F{i}"
        f_i_plus_1_col = f"F{i+1}"

        f_i_contracts = contract_chain[f_i_col].to_numpy(dtype=object)
        f_i_plus_1_contracts = contract_chain[f_i_plus_1_col].to_numpy(dtype=object)

        f_i_indices = np.array([contract_index.get(c, -1) for c in f_i_contracts])
        f_i_plus_1_indices = np.array([contract_index.get(c, -1) for c in f_i_plus_1_contracts])

        f_i_prices = np.full(len(values), np.nan)
        f_i_plus_1_prices = np.full(len(values), np.nan)

        valid_f_i = f_i_indices >= 0
        valid_f_i_plus_1 = f_i_plus_1_indices >= 0

        if valid_f_i.any():
            row_idx = np.nonzero(valid_f_i)[0]
            f_i_prices[valid_f_i] = values[row_idx, f_i_indices[valid_f_i]]
        if valid_f_i_plus_1.any():
            row_idx = np.nonzero(valid_f_i_plus_1)[0]
            f_i_plus_1_prices[valid_f_i_plus_1] = values[row_idx, f_i_plus_1_indices[valid_f_i_plus_1]]

        spread = f_i_plus_1_prices - f_i_prices
        result_dict[f"S{i}"] = spread

    result = pd.DataFrame(result_dict, index=panel.index)
    return result


def compute_liquidity_signal(
    panel: pd.DataFrame,
    front_next: pd.DataFrame,
    *,
    volume_field: str = "volume",
    alpha: float = 0.8,
    confirm: int = 1,
    meta_namespace: str = "meta",
) -> pd.Series:
    """Detect liquidity roll signals where next volume exceeds alpha * front volume."""
    contracts = [
        contract
        for contract in panel.columns.get_level_values(0).unique()
        if contract != meta_namespace
    ]
    if not contracts:
        raise ValueError("Panel does not contain contract columns")

    volume_df = _extract_field(panel, contracts, volume_field)
    volume_values = volume_df.to_numpy(dtype=float)

    contract_index = {contract: idx for idx, contract in enumerate(contracts)}
    front_idx = np.array([contract_index.get(c, -1) for c in front_next["front_contract"]], dtype=int)
    next_idx = np.array([contract_index.get(c, -1) for c in front_next["next_contract"]], dtype=int)

    signals = np.zeros(len(panel), dtype=bool)
    valid_mask = (front_idx >= 0) & (next_idx >= 0)
    if valid_mask.any():
        front_vol = volume_values[valid_mask, front_idx[valid_mask]]
        next_vol = volume_values[valid_mask, next_idx[valid_mask]]
        signals[valid_mask] = (next_vol >= alpha * front_vol) & (front_vol > 0)

    series = pd.Series(signals, index=panel.index, name="liquidity_roll")
    if confirm > 1:
        rolling = series.rolling(confirm).apply(lambda x: np.all(x == 1), raw=True)
        series = rolling.astype(bool)
    return series


def extract_front_next_volumes(
    panel: pd.DataFrame,
    front_next: pd.DataFrame,
    *,
    volume_field: str = "volume",
    meta_namespace: str = "meta",
) -> tuple[pd.Series, pd.Series]:
    """
    Extract F1 (front) and F2 (next) volume series from panel.

    Parameters
    ----------
    panel:
        DataFrame with MultiIndex columns ``(contract, field)``.
    front_next:
        DataFrame with 'front_contract' and 'next_contract' columns from identify_front_next().
    volume_field:
        Field name to extract (default ``"volume"``).
    meta_namespace:
        Name of the metadata column namespace inside the panel.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (front_volumes, next_volumes) as pandas Series indexed by panel.index
    """
    contracts = [
        contract
        for contract in panel.columns.get_level_values(0).unique()
        if contract != meta_namespace
    ]
    volume_df = _extract_field(panel, contracts, volume_field)
    volume_values = volume_df.to_numpy(dtype=float)

    contract_index = {contract: idx for idx, contract in enumerate(volume_df.columns)}

    front_idx = np.array(
        [contract_index.get(c, -1) for c in front_next["front_contract"]], dtype=int
    )
    next_idx = np.array([contract_index.get(c, -1) for c in front_next["next_contract"]], dtype=int)

    front_volumes = np.full(len(panel), np.nan)
    next_volumes = np.full(len(panel), np.nan)

    valid_front = front_idx >= 0
    valid_next = next_idx >= 0

    if valid_front.any():
        row_idx = np.nonzero(valid_front)[0]
        front_volumes[valid_front] = volume_values[row_idx, front_idx[valid_front]]

    if valid_next.any():
        row_idx = np.nonzero(valid_next)[0]
        next_volumes[valid_next] = volume_values[row_idx, next_idx[valid_next]]

    return (
        pd.Series(front_volumes, index=panel.index, name="front_volume"),
        pd.Series(next_volumes, index=panel.index, name="next_volume"),
    )


def _extract_field(panel: pd.DataFrame, contracts: Sequence[str], field: str) -> pd.DataFrame:
    if isinstance(panel.columns, pd.MultiIndex):
        tuples = [(contract, field) for contract in contracts if (contract, field) in panel.columns]
        if not tuples:
            return pd.DataFrame(index=panel.index)
        subset = panel.loc[:, tuples]
        subset.columns = [contract for contract, _ in subset.columns]
        return subset
    raise ValueError("Panel columns must be a MultiIndex (contract, field)")
