"""
Bucket-level panel assembly and manipulation for futures roll analysis.
Handles panel data with variable-granularity buckets instead of daily aggregation.
"""

from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from .bucket import get_bucket_info, BUCKET_DEFINITIONS


def assemble_bucket_panel(buckets_by_contract: Dict[str, pd.DataFrame], 
                          expiry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a wide panel with bucket-level granularity.
    
    Parameters:
    -----------
    buckets_by_contract : dict
        Dictionary mapping contract names to bucket-aggregated DataFrames
    expiry_df : pd.DataFrame
        DataFrame with columns: [contract, expiry_date, source, source_url]
        
    Returns:
    --------
    pd.DataFrame
        Wide panel with MultiIndex columns (contract, field, bucket) and 
        DatetimeIndex with multiple rows per date (one per bucket)
    """
    frames = []
    
    meta_cols = None
    for contract, dfb in buckets_by_contract.items():
        dfi = dfb.copy()
        
        # Attach expiry info (constant per contract)
        row = expiry_df.loc[expiry_df["contract"] == contract]
        exp = row["expiry_date"].iloc[0] if not row.empty else pd.NaT
        src = row["source"].iloc[0] if not row.empty else np.nan
        src_url = row["source_url"].iloc[0] if not row.empty else np.nan
        
        dfi["expiry"] = exp
        dfi["expiry_source"] = src
        dfi["expiry_source_url"] = src_url
        # capture meta columns from the first contract
        if meta_cols is None:
            cols = []
            for c in ["bucket", "bucket_label", "session"]:
                if c in dfi.columns:
                    cols.append(c)
            meta_cols = cols

        # Create MultiIndex columns with contract and field names
        dfi.columns = pd.MultiIndex.from_product([[contract], dfi.columns])
        frames.append(dfi)
    
    panel = pd.concat(frames, axis=1).sort_index()
    
    # Project meta columns (bucket metadata) into a ('meta', ...) namespace
    if meta_cols:
        for c in meta_cols:
            # Take meta from the first contract's column
            first_contract = panel.columns.get_level_values(0).unique().tolist()[0]
            panel[('meta', c)] = panel[(first_contract, c)]
        # Drop per-contract duplicates of meta cols
        for contract in panel.columns.get_level_values(0).unique():
            if contract == 'meta':
                continue
            for c in meta_cols:
                if (contract, c) in panel.columns:
                    panel = panel.drop(columns=[(contract, c)])
    
    return panel


def ensure_complete_buckets(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure each date has all 10 buckets represented in the panel.
    Fill missing buckets with NaN values.
    
    Parameters:
    -----------
    panel : pd.DataFrame
        Bucket panel data
        
    Returns:
    --------
    pd.DataFrame
        Panel with complete bucket coverage per date
    """
    # For now, return the panel as-is
    # The bucket structure is already embedded in the data from build_buckets_by_contract
    return panel


def mark_front_next_buckets(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Identify front and next contracts for each bucket period.
    
    Logic remains similar to daily version but operates at bucket granularity:
    - 0 active contracts → front=NaN, next=NaN
    - 1 active contract → front=that contract, next=NaN
    - 2+ active contracts → front=nearest expiry, next=second nearest
    
    Parameters:
    -----------
    panel : pd.DataFrame
        Bucket panel with contract data
        
    Returns:
    --------
    pd.DataFrame
        Panel with added ('meta','front_contract') and ('meta','next_contract') columns
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
        except Exception:
            expiry_map[c] = pd.NaT
    
    # Process each timestamp-bucket combination
    front_list = []
    next_list = []
    
    for idx in panel.index:
        # Handle both single index and MultiIndex
        if isinstance(panel.index, pd.MultiIndex):
            dt = idx[0] if isinstance(idx, tuple) else idx
        else:
            dt = idx
        
        dt_normalized = pd.Timestamp(dt).normalize()
        active = []
        
        for c in contracts:
            try:
                # Check if contract has a close price at this bucket
                close_price = panel.at[idx, (c, "close")]
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
    panel_out[('meta', 'front_contract')] = pd.Series(front_list, index=panel.index)
    panel_out[('meta', 'next_contract')] = pd.Series(next_list, index=panel.index)
    
    # Add bucket information to meta if not present
    if isinstance(panel.index, pd.MultiIndex) and 'bucket' in panel.index.names:
        panel_out[('meta', 'bucket')] = panel.index.get_level_values('bucket')
    
    # Validation
    front_filled = pd.Series(front_list).notna().sum()
    next_filled = pd.Series(next_list).notna().sum()
    next_only = sum(1 for f, n in zip(front_list, next_list) if pd.isna(f) and pd.notna(n))
    
    if next_only > 0:
        raise ValueError(f"Logic error: {next_only} bucket periods have next but no front!")
    
    print(f"Bucket front/next identification: front={front_filled}, next={next_filled}, both={min(front_filled, next_filled)}")
    
    return panel_out


def compute_bucket_spread(panel: pd.DataFrame, price_field: str = "close") -> pd.Series:
    """
    Calculate calendar spread (next - front) for each bucket.
    
    Parameters:
    -----------
    panel : pd.DataFrame
        Bucket panel with front/next contract identifiers
    price_field : str
        Price field to use for spread calculation
        
    Returns:
    --------
    pd.Series
        Spread values indexed by timestamp-bucket
    """
    f = panel[("meta", "front_contract")].astype("string")
    n = panel[("meta", "next_contract")].astype("string")
    
    vals = []
    for idx in panel.index:
        fc = f.loc[idx]
        nc = n.loc[idx]
        try:
            pf = panel.at[idx, (fc, price_field)] if pd.notna(fc) else float("nan")
            pn = panel.at[idx, (nc, price_field)] if pd.notna(nc) else float("nan")
            vals.append(pn - pf)
        except Exception:
            vals.append(float("nan"))
    
    return pd.Series(vals, index=panel.index, name=f"spread_{price_field}_next_minus_front")


def aggregate_buckets_to_daily(bucket_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bucket panel back to daily for validation and comparison.
    
    Parameters:
    -----------
    bucket_panel : pd.DataFrame
        Panel with bucket-level data
        
    Returns:
    --------
    pd.DataFrame
        Daily aggregated panel
    """
    # Extract date from index
    if isinstance(bucket_panel.index, pd.MultiIndex):
        dates = pd.to_datetime(bucket_panel.index.get_level_values(0)).date
    else:
        dates = pd.to_datetime(bucket_panel.index).date
    
    bucket_panel = bucket_panel.copy()
    bucket_panel['date'] = dates
    
    # Define aggregation rules
    agg_rules = {}
    
    for col in bucket_panel.columns:
        if isinstance(col, tuple):
            contract, field = col[0], col[1]
            if field in ['open']:
                # First value of the day
                agg_rules[col] = 'first'
            elif field in ['high']:
                # Maximum of the day
                agg_rules[col] = 'max'
            elif field in ['low']:
                # Minimum of the day
                agg_rules[col] = 'min'
            elif field in ['close']:
                # Last value of the day
                agg_rules[col] = 'last'
            elif field in ['volume']:
                # Sum of all buckets
                agg_rules[col] = 'sum'
            elif field in ['open_interest']:
                # Last value of the day
                agg_rules[col] = 'last'
            elif field in ['expiry', 'expiry_source', 'expiry_source_url']:
                # Constant values - take first
                agg_rules[col] = 'first'
            elif field in ['front_contract', 'next_contract']:
                # Most common value (mode) or last
                agg_rules[col] = lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[-1]
            else:
                # For non-numeric, use first; for numeric unknowns, mean
                agg_rules[col] = (lambda s: s.first_valid_index() and s.dropna().iloc[0]) if bucket_panel[col].dtype == 'O' else 'mean'
    
    # Perform aggregation
    daily_panel = bucket_panel.groupby('date').agg(agg_rules)
    daily_panel.index = pd.DatetimeIndex(daily_panel.index)
    
    return daily_panel


def calculate_bucket_liquidity_transitions(panel: pd.DataFrame, 
                                          alpha: float = 0.8,
                                          price_field: str = "close") -> pd.Series:
    """
    Detect liquidity transitions at bucket level where next volume >= alpha * front volume.
    
    Parameters:
    -----------
    panel : pd.DataFrame
        Bucket panel with volume data
    alpha : float
        Threshold for volume ratio (default: 0.8)
    price_field : str
        Price field to verify contract activity
        
    Returns:
    --------
    pd.Series
        Boolean series indicating liquidity roll signal per bucket
    """
    f = panel[('meta', 'front_contract')].astype('string')
    n = panel[('meta', 'next_contract')].astype('string')
    
    def safe_get(contract: str, idx, field: str):
        try:
            return panel.at[idx, (contract, field)]
        except Exception:
            return np.nan
    
    signals = []
    for idx in panel.index:
        fc = f.loc[idx]
        nc = n.loc[idx]
        
        if pd.isna(fc) or pd.isna(nc):
            signals.append(False)
            continue
        
        vf = safe_get(fc, idx, 'volume')
        vn = safe_get(nc, idx, 'volume')
        
        if pd.isna(vf) or pd.isna(vn) or vf <= 0:
            signals.append(False)
        else:
            signals.append(bool(vn >= alpha * vf))
    
    return pd.Series(signals, index=panel.index, name='bucket_liquidity_roll')
