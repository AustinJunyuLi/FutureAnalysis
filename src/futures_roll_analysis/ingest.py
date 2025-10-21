from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Pattern, Sequence, Tuple

import numpy as np
import pandas as pd

from .buckets import aggregate_to_buckets


DEFAULT_MONTH_CODES: Dict[str, int] = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}


def build_contract_regex(root_symbol: str) -> Pattern[str]:
    """Build a regex matcher for contract filenames belonging to a given root symbol."""
    return re.compile(
        rf"{re.escape(root_symbol)}_?([FGHJKMNQUVXZ])_?(\d{{1,2}}|\d{{4}})",
        flags=re.IGNORECASE,
    )


def normalize_contract_code(
    contract_raw: str,
    root_symbol: str,
    regex: Optional[Pattern[str]] = None,
) -> Optional[str]:
    """Normalise contract tokens like ``HGZ25`` into canonical ``HGZ2025`` form."""
    pattern = regex or build_contract_regex(root_symbol)
    match = pattern.search(contract_raw)
    if not match:
        return None

    month_code = match.group(1).upper()
    year_fragment = match.group(2)

    if len(year_fragment) == 2:
        year = int(year_fragment)
        year = 2000 + year if year <= 79 else 1900 + year
    else:
        year = int(year_fragment)

    return f"{root_symbol.upper()}{month_code}{year:04d}"


def find_contract_files(
    root: Path,
    root_symbol: str,
    extensions: Sequence[str] = ("csv", "txt", "parquet", "pq"),
    regex: Optional[Pattern[str]] = None,
) -> List[Path]:
    """Recursively discover minute files for the specified contract root symbol."""
    root = root.expanduser().resolve()
    pattern = regex or build_contract_regex(root_symbol)
    matches: List[Path] = []

    for extension in extensions:
        ext = extension.lstrip(".")
        for path in root.rglob(f"*{root_symbol.upper()}*.{ext}"):
            if path.is_file() and pattern.search(path.name):
                matches.append(path)

    matches.sort()
    return matches


def load_minutes(
    file_path: Path,
    required_columns: Iterable[str] = ("open", "high", "low", "close", "volume"),
) -> pd.DataFrame:
    """Load a minute-level file (CSV/Parquet) into a DataFrame indexed by timestamp."""
    file_path = file_path.resolve()
    if file_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
        first_col = df.columns[0]
        try:
            pd.to_datetime(first_col)
            df = pd.read_csv(
                file_path,
                header=None,
                names=["timestamp", "open", "high", "low", "close", "volume"],
            )
        except (ValueError, TypeError):
            pass

    ts_col = _detect_timestamp_column(df.columns)
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {file_path}")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=False, errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.set_index(ts_col).sort_index()

    rename_map: Dict[str, str] = {}
    for col in df.columns:
        lowered = col.lower()
        if lowered in {"o", "open_price"}:
            rename_map[col] = "open"
        elif lowered in {"h", "high_price"}:
            rename_map[col] = "high"
        elif lowered in {"l", "low_price"}:
            rename_map[col] = "low"
        elif lowered in {"c", "close_price", "last"}:
            rename_map[col] = "close"
        elif lowered in {"vol", "volume"}:
            rename_map[col] = "volume"
        elif lowered in {"oi", "openinterest", "open_interest"}:
            rename_map[col] = "open_interest"

    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {file_path}")

    return df


def minutes_to_daily(df_minutes: pd.DataFrame, tz: str = "US/Central") -> pd.DataFrame:
    """Aggregate minute OHLCV data to daily bars."""
    if not isinstance(df_minutes.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must be indexed by timestamps")

    if df_minutes.index.tz is None:
        localised = df_minutes.tz_localize(tz)
    else:
        localised = df_minutes.tz_convert(tz)

    daily = pd.DataFrame(
        {
            "open": localised["open"].resample("1D").first(),
            "high": localised["high"].resample("1D").max(),
            "low": localised["low"].resample("1D").min(),
            "close": localised["close"].resample("1D").last(),
            "volume": localised["volume"].resample("1D").sum(),
        }
    )
    if "open_interest" in localised.columns:
        daily["open_interest"] = localised["open_interest"].resample("1D").last()

    daily = daily.dropna(how="all")
    daily.index = daily.index.tz_localize(None)
    return daily


def minutes_to_buckets(df_minutes: pd.DataFrame, tz: str = "US/Central") -> pd.DataFrame:
    """Aggregate minute OHLCV data to variable-granularity buckets."""
    return aggregate_to_buckets(df_minutes, tz=tz)


def build_contract_frames(
    files: Iterable[Path],
    *,
    root_symbol: str,
    tz: str,
    aggregate: str,
    regex: Optional[Pattern[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load all discovered files, normalise contract codes, aggregate, and group by contract.

    Parameters
    ----------
    files:
        Iterable of discovered minute-data files.
    root_symbol:
        Futures root symbol (e.g., ``HG``).
    tz:
        Timezone used for aggregation.
    aggregate:
        ``"daily"`` or ``"bucket"``.
    regex:
        Optional custom contract regex for normalisation.
    """
    aggregator = {
        "daily": minutes_to_daily,
        "bucket": minutes_to_buckets,
    }.get(aggregate)

    if aggregator is None:
        raise ValueError(f"Unsupported aggregate mode: {aggregate}")

    pattern = regex or build_contract_regex(root_symbol)
    grouped: Dict[str, pd.DataFrame] = {}

    for path in files:
        contract = normalize_contract_code(path.name, root_symbol=root_symbol, regex=pattern)
        if not contract:
            continue
        try:
            minutes = load_minutes(path)
            aggregated = aggregator(minutes, tz=tz)
        except Exception as exc:  # pragma: no cover - defensive logging hook
            print(f"Warning: failed to process {path}: {exc}")
            continue

        if contract in grouped:
            combined = pd.concat([grouped[contract], aggregated]).sort_index()
            agg_map = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            if "open_interest" in combined.columns:
                agg_map["open_interest"] = "last"
            grouped[contract] = combined.groupby(level=0).agg(agg_map)
        else:
            grouped[contract] = aggregated

    return grouped


def _detect_timestamp_column(columns: Iterable[str]) -> Optional[str]:
    candidates = [
        "timestamp",
        "datetime",
        "date_time",
        "time",
        "dt",
        "DateTime",
        "Timestamp",
    ]
    columns_list = list(columns)
    for candidate in candidates:
        if candidate in columns_list:
            return candidate

    lower_map = {col.lower(): col for col in columns_list}
    for candidate in candidates:
        lowered = candidate.lower()
        if lowered in lower_map:
            return lower_map[lowered]
    return None
