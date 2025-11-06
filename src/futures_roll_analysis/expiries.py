from __future__ import annotations

"""
Expiry metadata utilities.

Provides a typed loader that parses per-contract expiry timestamps as
timezone-aware UTC instants for use in deterministic label selection.
"""

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class ExpirySpec:
    """Container for canonical expiry timestamps in UTC.

    Attributes
    ----------
    expiry_ts_utc:
        Mapping from contract code to UTC expiry timestamp.
    rule:
        Provenance label (e.g., "LTD_LOCAL->UTC" or a product rule identifier).
    tz_exchange:
        Exchange local timezone name (e.g., "America/Chicago").
    """

    expiry_ts_utc: Mapping[str, pd.Timestamp]
    rule: str
    tz_exchange: str


def load_expiries(path: str, tz_exchange: str, *, contract_col: str = "contract", local_iso_col: str = "expiry_local_iso", rule: str = "LTD_LOCAL->UTC") -> ExpirySpec:
    """
    Load contract expiries from a CSV with local-time ISO strings and convert to UTC.

    Parameters
    ----------
    path:
        CSV path containing contract and local ISO expiry timestamp columns.
    tz_exchange:
        IANA timezone string for the exchange local time (e.g., "America/Chicago").
    contract_col:
        Column name for contract code (default "contract").
    local_iso_col:
        Column with local ISO timestamps (default "expiry_local_iso").
    rule:
        Provenance label to attach to the returned spec.
    """
    df = pd.read_csv(path)
    if contract_col not in df.columns or local_iso_col not in df.columns:
        raise ValueError(f"CSV must include '{contract_col}' and '{local_iso_col}' columns")

    local = pd.to_datetime(df[local_iso_col], utc=False, errors="raise").dt.tz_localize(tz_exchange, ambiguous="raise", nonexistent="raise")
    utc = local.dt.tz_convert("UTC")
    mapping = dict(zip(df[contract_col].astype(str), utc))
    return ExpirySpec(expiry_ts_utc=mapping, rule=rule, tz_exchange=tz_exchange)

