from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def build_expiry_map(expiry_df: pd.DataFrame) -> pd.Series:
    """Return a Series mapping contract codes to expiry timestamps."""
    if "contract" not in expiry_df.columns or "expiry_date" not in expiry_df.columns:
        raise ValueError("Expiry metadata must include 'contract' and 'expiry_date' columns")
    series = (
        expiry_df.drop_duplicates("contract")
        .set_index("contract")["expiry_date"]
        .pipe(pd.to_datetime)
        .dt.normalize()
    )
    return series


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
    ).normalize()

    close_values = close_df.to_numpy(dtype=float)
    available = np.isfinite(close_values)

    expiry_array = pd.to_datetime(expiry_series).to_numpy(dtype="datetime64[ns]")
    expiry_int = expiry_array.astype("datetime64[ns]").astype("int64")
    date_int = dates.to_numpy(dtype="datetime64[ns]").astype("int64")

    delta = expiry_int.reshape(1, -1) - date_int.reshape(-1, 1)
    active_mask = available & (delta >= 0)

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


def _extract_field(panel: pd.DataFrame, contracts: Sequence[str], field: str) -> pd.DataFrame:
    if isinstance(panel.columns, pd.MultiIndex):
        tuples = [(contract, field) for contract in contracts if (contract, field) in panel.columns]
        if not tuples:
            return pd.DataFrame(index=panel.index)
        subset = panel.loc[:, tuples]
        subset.columns = [contract for contract, _ in subset.columns]
        return subset
    raise ValueError("Panel columns must be a MultiIndex (contract, field)")
