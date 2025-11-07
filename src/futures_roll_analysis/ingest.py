from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Pattern, Sequence, Tuple

import numpy as np
import pandas as pd

import logging
from .buckets import aggregate_to_buckets

LOGGER = logging.getLogger(__name__)


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
        rf"{re.escape(root_symbol)}_?([FGHJKMNQUVXZ])_?(\d{{4}}|\d{{1,2}})",
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

    if len(year_fragment) <= 2:
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
    *,
    timestamp_format: Optional[str] = None,
) -> pd.DataFrame:
    """Load a minute-level file (CSV/Parquet) into a DataFrame indexed by timestamp."""
    file_path = file_path.resolve()
    suffix = file_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(file_path)
        ts_col = _detect_timestamp_column(df.columns)
        if ts_col is None:
            raise ValueError(f"No timestamp column found in {file_path}")
    else:
        df = pd.read_csv(file_path)
        ts_col = _detect_timestamp_column(df.columns)
        if ts_col is None:
            # Try reading without headers if no timestamp column found
            df = pd.read_csv(file_path, header=None)
            df, ts_col = _coerce_minute_dataframe(df)

    parse_kwargs = {"errors": "coerce"}
    if timestamp_format:
        parse_kwargs["format"] = timestamp_format
    df[ts_col] = pd.to_datetime(df[ts_col], utc=False, **parse_kwargs)
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
    timestamp_format: Optional[str] = None,
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
            minutes = load_minutes(path, timestamp_format=timestamp_format)
            aggregated = aggregator(minutes, tz=tz)
        except Exception as exc:  # pragma: no cover - defensive logging hook
            LOGGER.warning("Failed to process %s: %s", path, exc)
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


def _coerce_minute_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Infer sensible column names for minute data lacking explicit headers.

    Returns
    -------
    Tuple of coerced DataFrame and resolved timestamp column name.
    """
    columns = list(df.columns)
    default_names = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    ]

    if all(isinstance(col, (int, np.integer)) for col in columns):
        rename = {
            col: default_names[i] if i < len(default_names) else f"extra_{i}"
            for i, col in enumerate(columns)
        }
        df = df.rename(columns=rename)
        timestamp_col = rename[columns[0]]
    else:
        first_col = columns[0]
        parsed = pd.to_datetime(df[first_col], errors="coerce")
        if parsed.notna().sum() == 0:
            raise ValueError("Unable to infer timestamp column for minute data without headers")
        df[first_col] = parsed
        if isinstance(first_col, str):
            timestamp_col = first_col
        else:
            timestamp_col = "timestamp"
            df = df.rename(columns={first_col: timestamp_col})

    return df, timestamp_col


def _detect_timestamp_column(columns: Iterable[object]) -> Optional[str]:
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
    for col in columns_list:
        if isinstance(col, str) and col in candidates:
            return col

    lower_map = {col.lower(): col for col in columns_list if isinstance(col, str)}
    for candidate in candidates:
        lowered = candidate.lower()
        if lowered in lower_map:
            return lower_map[lowered]
    return None
