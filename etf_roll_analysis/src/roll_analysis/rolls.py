from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


def mark_front_next(panel: pd.DataFrame) -> pd.DataFrame:
    """Adds ('meta','front_contract') and ('meta','next_contract') columns based on nearest expiry.
    
    Logic:
    - 0 active contracts → front=NaN, next=NaN
    - 1 active contract → front=that contract, next=NaN
    - 2+ active contracts → front=nearest expiry, next=second nearest
    """
    # Get all contracts, excluding 'meta' if present
    all_contracts = panel.columns.get_level_values(0).unique().tolist()
    contracts = [c for c in all_contracts if c != 'meta']

    # Build expiry map (constant per contract)
    expiry_map: Dict[str, pd.Timestamp] = {}
    for c in contracts:
        try:
            exp_series = panel[(c, "expiry")].dropna()
            if len(exp_series) > 0:
                expiry_map[c] = pd.to_datetime(exp_series.iloc[0]).normalize()
            else:
                expiry_map[c] = pd.NaT
        except Exception as e:
            expiry_map[c] = pd.NaT

    # Identify front and next for each date
    idx = panel.index
    front_list = []
    next_list = []
    
    for dt in idx:
        dt_normalized = pd.Timestamp(dt).normalize()
        active = []
        
        for c in contracts:
            try:
                # Check if contract has a close price on this date
                close_price = panel.at[dt, (c, "close")]
                if pd.notna(close_price):
                    exp_date = expiry_map.get(c, pd.NaT)
                    # Only include if expiry is valid and in the future
                    if pd.notna(exp_date) and exp_date >= dt_normalized:
                        active.append((c, exp_date))
            except (KeyError, Exception):
                continue
        
        # Determine front and next based on number of active contracts
        if len(active) == 0:
            front_list.append(np.nan)
            next_list.append(np.nan)
        elif len(active) == 1:
            front_list.append(active[0][0])
            next_list.append(np.nan)
        else:  # 2 or more
            active_sorted = sorted(active, key=lambda x: x[1])
            front_list.append(active_sorted[0][0])
            next_list.append(active_sorted[1][0])
    
    # Create output panel with meta columns
    panel_out = panel.copy()
    panel_out[('meta', 'front_contract')] = pd.Series(front_list, index=idx)
    panel_out[('meta', 'next_contract')] = pd.Series(next_list, index=idx)
    
    # Validation
    front_filled = pd.Series(front_list).notna().sum()
    next_filled = pd.Series(next_list).notna().sum()
    next_only = sum(1 for f, n in zip(front_list, next_list) if pd.isna(f) and pd.notna(n))
    
    if next_only > 0:
        raise ValueError(f"Logic error: {next_only} days have next but no front!")
    
    print(f"Front/next identification: front={front_filled}, next={next_filled}, both={min(front_filled, next_filled)}")
    
    return panel_out


def compute_fixed_roll_dates(expiry_series: pd.Series, N: int = 5) -> pd.Series:
    """Return a per-contract fixed roll date = expiry - N business days."""
    return (pd.to_datetime(expiry_series).dt.normalize() - BDay(N)).dt.normalize()


def liquidity_roll_signal(panel: pd.DataFrame, alpha: float = 0.8, confirm_days: int = 1,
                          price_field: str = "close") -> pd.Series:
    """Boolean Series over index dates where next volume >= alpha * front volume.
    Optionally requires confirm_days consecutive True values.
    """
    idx = panel.index
    f = panel[('meta','front_contract')].astype('string')
    n = panel[('meta','next_contract')].astype('string')

    def safe_get(contract: str, dt, field: str):
        try:
            return panel.at[dt, (contract, field)]
        except Exception:
            return np.nan

    raw = []
    for dt in idx:
        fc = f.loc[dt]
        nc = n.loc[dt]
        if pd.isna(fc) or pd.isna(nc):
            raw.append(False)
            continue
        vf = safe_get(fc, dt, 'volume')
        vn = safe_get(nc, dt, 'volume')
        if pd.isna(vf) or pd.isna(vn) or vf <= 0:
            raw.append(False)
        else:
            raw.append(bool(vn >= alpha * vf))

    sig = pd.Series(raw, index=idx, name='liquidity_roll')
    if confirm_days > 1:
        sig = sig.rolling(confirm_days).apply(lambda x: 1.0 if np.all(x == 1) else 0.0, raw=True).astype(bool)
    return sig
