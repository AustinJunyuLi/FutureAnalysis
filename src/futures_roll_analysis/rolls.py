from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .labeler import compute_strip_labels


def _extract_front_next_field(
    panel: pd.DataFrame,
    front_next: pd.DataFrame,
    *,
    field: str,
    meta_namespace: str = "meta",
) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Return front/next series for the requested field if available."""

    contracts = [c for c in panel.columns.get_level_values(0).unique() if c != meta_namespace]
    field_df = _extract_field(panel, contracts, field)
    if field_df.empty:
        return None, None

    values = field_df.to_numpy(dtype=float)
    contract_index = {contract: idx for idx, contract in enumerate(field_df.columns)}

    front_idx = np.array([contract_index.get(c, -1) for c in front_next["front_contract"]], dtype=int)
    next_idx = np.array([contract_index.get(c, -1) for c in front_next["next_contract"]], dtype=int)

    front_values = np.full(len(panel), np.nan)
    next_values = np.full(len(panel), np.nan)

    valid_front = front_idx >= 0
    if valid_front.any():
        row_idx = np.nonzero(valid_front)[0]
        front_values[valid_front] = values[row_idx, front_idx[valid_front]]

    valid_next = next_idx >= 0
    if valid_next.any():
        row_idx = np.nonzero(valid_next)[0]
        next_values[valid_next] = values[row_idx, next_idx[valid_next]]

    return (
        pd.Series(front_values, index=panel.index, name=f"front_{field}"),
        pd.Series(next_values, index=panel.index, name=f"next_{field}"),
    )

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


def identify_front_to_f12(
    panel: pd.DataFrame,
    expiry_map: pd.Series,
    *,
    tz_exchange: str = "America/Chicago",
    max_contracts: int = 12,
    meta_namespace: str = "meta",
) -> pd.DataFrame:
    """
    Deterministic identification of F1..F{max_contracts} using expiry timestamps only.

    Uses a strict expiry-based approach where contracts switch at their exact expiry instant.
    Converts the panel index to UTC (assuming exchange-local tz for naive indices)
    and expiries to UTC, then uses compute_strip_labels to build the strip.

    Parameters
    ----------
    panel:
        DataFrame with MultiIndex columns ``(contract, field)``.
    expiry_map:
        Series mapping contracts to expiry ``Timestamp`` objects.
    tz_exchange:
        Exchange timezone (default "America/Chicago" for CME).
    max_contracts:
        Number of contract levels to identify (default 12 for F1...F12).
    meta_namespace:
        Name of the metadata column namespace inside the panel.

    Returns
    -------
    DataFrame with columns ['F1', 'F2', ..., 'F{max_contracts}']
    """
    ts = panel.index.get_level_values(0) if isinstance(panel.index, pd.MultiIndex) else panel.index
    idx = pd.DatetimeIndex(ts)

    # Ensure index is monotonic and handle tz-awareness
    if not idx.is_monotonic_increasing:
        idx = idx.sort_values()

    if idx.tz is None:
        idx_local = idx.tz_localize(tz_exchange, nonexistent="shift_forward", ambiguous="infer")
    else:
        idx_local = idx.tz_convert(tz_exchange)

    idx_utc = idx_local.tz_convert("UTC")

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
            # For individual timestamps, use True (treat as DST) for ambiguous times
            # Expiry times are typically 17:00, not in 1:00-2:00 AM ambiguous window
            ts_local = ts_local.tz_localize(tz_exchange, ambiguous=True, nonexistent="shift_forward")
        expiries_utc[c] = ts_local.tz_convert("UTC")

    strip = compute_strip_labels(idx_utc, contracts_sorted, expiries_utc, depth=max_contracts)
    strip.index = panel.index
    return strip


def identify_front_next(
    panel: pd.DataFrame,
    expiry_map: pd.Series,
    *,
    tz_exchange: str = "America/Chicago",
    meta_namespace: str = "meta",
) -> pd.DataFrame:
    """
    Deterministic front/next (F1/F2) identification using expiry timestamps only.

    Wrapper over identify_front_to_f12 for the common F1/F2 use case.
    Contracts switch at their exact expiry instant.

    Parameters
    ----------
    panel:
        DataFrame with MultiIndex columns ``(contract, field)``.
    expiry_map:
        Series mapping contracts to expiry ``Timestamp`` objects.
    tz_exchange:
        Exchange timezone (default "America/Chicago" for CME).
    meta_namespace:
        Name of the metadata column namespace inside the panel.

    Returns
    -------
    DataFrame with columns ['front_contract', 'next_contract']
    """
    strip = identify_front_to_f12(
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
    front_series, next_series = _extract_front_next_field(
        panel, front_next, field=volume_field, meta_namespace=meta_namespace
    )
    if front_series is None or next_series is None:
        return pd.Series(False, index=panel.index, name="liquidity_roll")

    signals = (next_series >= alpha * front_series) & (front_series > 0)
    signals = signals.fillna(False)
    series = signals.rename("liquidity_roll")
    if confirm > 1:
        rolling = series.rolling(confirm).apply(lambda x: np.all(x == 1), raw=True)
        series = rolling.eq(1.0).fillna(False)
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
    front_series, next_series = _extract_front_next_field(
        panel, front_next, field=volume_field, meta_namespace=meta_namespace
    )
    if front_series is None or next_series is None:
        empty = pd.Series(np.nan, index=panel.index)
        return empty.rename("front_volume"), empty.rename("next_volume")
    return front_series.rename("front_volume"), next_series.rename("next_volume")


def compute_open_interest_signal(
    panel: pd.DataFrame,
    front_next: pd.DataFrame,
    *,
    oi_field: str = "open_interest",
    ratio: float = 0.75,
    confirm: int = 1,
    meta_namespace: str = "meta",
) -> Optional[pd.Series]:
    """Detect open-interest migrations from front to next contracts."""

    front_series, next_series = _extract_front_next_field(
        panel, front_next, field=oi_field, meta_namespace=meta_namespace
    )
    if front_series is None or next_series is None:
        return None

    signals = (next_series >= ratio * front_series) & (front_series > 0)
    signals = signals.fillna(False)
    series = signals.rename("open_interest_roll")
    if confirm > 1:
        rolling = series.rolling(confirm).apply(lambda x: np.all(x == 1), raw=True)
        series = rolling.eq(1.0).fillna(False)
    return series


def _extract_field(panel: pd.DataFrame, contracts: Sequence[str], field: str) -> pd.DataFrame:
    if isinstance(panel.columns, pd.MultiIndex):
        tuples = [(contract, field) for contract in contracts if (contract, field) in panel.columns]
        if not tuples:
            return pd.DataFrame(index=panel.index)
        subset = panel.loc[:, tuples]
        subset.columns = [contract for contract, _ in subset.columns]
        return subset
    raise ValueError("Panel columns must be a MultiIndex (contract, field)")
