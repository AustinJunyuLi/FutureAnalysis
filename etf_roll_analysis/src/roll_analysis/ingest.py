import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Month codes mapping for futures
MONTH_CODE_TO_NUM = {
    "F": 1,  # Jan
    "G": 2,  # Feb
    "H": 3,  # Mar
    "J": 4,  # Apr
    "K": 5,  # May
    "M": 6,  # Jun
    "N": 7,  # Jul
    "Q": 8,  # Aug
    "U": 9,  # Sep
    "V": 10, # Oct
    "X": 11, # Nov
    "Z": 12, # Dec
}

CONTRACT_REGEX = re.compile(r"HG_?([FGHJKMNQUVXZ])_?(\d{1,2}|\d{4})", re.IGNORECASE)


def normalize_contract_code(contract_raw: str) -> Optional[str]:
    """Normalize strings like 'HGZ25' or 'hgZ2025' to 'HGZ2025'."""
    m = CONTRACT_REGEX.search(contract_raw)
    if not m:
        return None
    mcode = m.group(1).upper()
    y = m.group(2)
    # Normalize year to 4 digits (assume <=79 -> 2000s, else 1900s)
    if len(y) == 2:
        yi = int(y)
        year = 2000 + yi if yi <= 79 else 1900 + yi
    else:
        year = int(y)
    return f"HG{mcode}{year:04d}"


def find_hg_files(root: str, extensions: Tuple[str, ...] = ("csv", "txt", "parquet", "pq")) -> List[Path]:
    """Recursively find files that likely contain HG minute data."""
    paths: List[Path] = []
    root_path = Path(root)
    for ext in extensions:
        # broad search for files containing 'HG' and month code
        pat = f"**/*HG*.[{ext}]" if len(ext) == 1 else f"**/*HG*.{ext}"
        for p in root_path.glob(pat):
            if p.is_file():
                # Verify a contract-like token exists in the filename
                if CONTRACT_REGEX.search(p.name):
                    paths.append(p)
    return sorted(set(paths))


def _detect_timestamp_column(cols: List[str]) -> Optional[str]:
    candidates = [
        "timestamp", "datetime", "date_time", "time", "dt", "DateTime", "Timestamp",
    ]
    for c in candidates:
        if c in cols:
            return c
    # case-insensitive match
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_minutes(fp: Path) -> pd.DataFrame:
    """Load a minute-level file (csv or parquet). Returns tz-naive or tz-aware index DataFrame."""
    if fp.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(fp)
    else:
        # Try reading with auto-detected header first
        df = pd.read_csv(fp)
        # If first column looks like a timestamp value (not a name), assume no header
        first_col = df.columns[0]
        try:
            pd.to_datetime(first_col)
            # It's a timestamp, not a column name - reload without header
            df = pd.read_csv(fp, header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        except (ValueError, TypeError):
            pass  # Has headers, continue
    # Find timestamp column
    ts_col = _detect_timestamp_column(df.columns.tolist())
    if ts_col is None:
        raise ValueError(f"No timestamp-like column found in {fp}")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=False, errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.set_index(ts_col).sort_index()

    # Normalize column names
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"o", "open_price"}: rename_map[c] = "open"
        elif lc in {"h", "high_price"}: rename_map[c] = "high"
        elif lc in {"l", "low_price"}: rename_map[c] = "low"
        elif lc in {"c", "close_price", "last"}: rename_map[c] = "close"
        elif lc in {"vol", "volume"}: rename_map[c] = "volume"
        elif lc in {"oi", "openinterest", "open_interest"}: rename_map[c] = "open_interest"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {fp}")

    return df


def minutes_to_daily(df_minute: pd.DataFrame, tz: str = "US/Central") -> pd.DataFrame:
    """Aggregate minute data to daily OHLCV (and OI if present), localized/converted to tz."""
    idx = df_minute.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")
    if idx.tz is None:
        # Assume timestamps are in tz already; localize
        dfl = df_minute.tz_localize(tz)
    else:
        dfl = df_minute.tz_convert(tz)

    daily = pd.DataFrame({
        "open": dfl["open"].resample("1D").first(),
        "high": dfl["high"].resample("1D").max(),
        "low": dfl["low"].resample("1D").min(),
        "close": dfl["close"].resample("1D").last(),
        "volume": dfl["volume"].resample("1D").sum(),
    })
    if "open_interest" in dfl.columns:
        daily["open_interest"] = dfl["open_interest"].resample("1D").last()
    daily = daily.dropna(how="all")
    daily.index = daily.index.tz_localize(None)
    return daily


def build_daily_by_contract(paths: List[Path], tz: str = "US/Central") -> Dict[str, pd.DataFrame]:
    """Load all files, group by normalized contract code, and aggregate to daily."""
    out: Dict[str, pd.DataFrame] = {}
    for fp in paths:
        contract = normalize_contract_code(fp.name)
        if not contract:
            continue
        try:
            dfm = load_minutes(fp)
            dfd = minutes_to_daily(dfm, tz=tz)
            if contract in out:
                # combine duplicates (multiple files per contract)
                concat_df = pd.concat([out[contract], dfd]).sort_index()
                agg_map = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                if "open_interest" in concat_df.columns:
                    agg_map["open_interest"] = "last"
                out[contract] = concat_df.groupby(level=0).agg(agg_map)
            else:
                out[contract] = dfd
        except Exception as e:
            print(f"Warning: failed to load {fp}: {e}")
    return out


def minutes_to_buckets(df_minute: pd.DataFrame, tz: str = "US/Central") -> pd.DataFrame:
    """
    Aggregate minute data to variable-granularity buckets.
    Uses hourly buckets for US regular hours and broader sessions for off-peak.
    
    Parameters:
    -----------
    df_minute : pd.DataFrame
        Minute-level OHLCV data
    tz : str
        Timezone for aggregation (default: US/Central)
        
    Returns:
    --------
    pd.DataFrame
        Bucket-aggregated data with bucket metadata
    """
    from .bucket import aggregate_to_buckets
    return aggregate_to_buckets(df_minute, tz=tz)


def build_buckets_by_contract(paths: List[Path], tz: str = "US/Central") -> Dict[str, pd.DataFrame]:
    """Load all files, group by normalized contract code, and aggregate to buckets."""
    out: Dict[str, pd.DataFrame] = {}
    for fp in paths:
        contract = normalize_contract_code(fp.name)
        if not contract:
            continue
        try:
            dfm = load_minutes(fp)
            dfb = minutes_to_buckets(dfm, tz=tz)
            if contract in out:
                # combine duplicates (multiple files per contract)
                # Simply concatenate and remove duplicates based on index
                out[contract] = pd.concat([out[contract], dfb]).sort_index()
                # Remove exact duplicates
                out[contract] = out[contract][~out[contract].index.duplicated(keep='first')]
            else:
                out[contract] = dfb
        except Exception as e:
            print(f"Warning: failed to load {fp}: {e}")
    return out
