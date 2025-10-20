from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

MONTH_CODE_TO_NUM = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5,
    "M": 6, "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

@dataclass
class ExpiryInfo:
    contract: str
    expiry_date: pd.Timestamp
    source: str
    source_url: Optional[str] = None


def derive_expiry_from_data(contract: str, dfd: pd.DataFrame) -> Optional[ExpiryInfo]:
    """Use the last date with non-null close (and optionally volume>0) as a proxy for last trade date.
    This is a robust fallback when official metadata is unavailable.
    """
    if dfd.empty:
        return None
    # Prefer the last date where volume > 0; otherwise last date with any data
    if "volume" in dfd.columns:
        nonzero = dfd[dfd["volume"].fillna(0) > 0]
        last_date = (nonzero.index.max() if not nonzero.empty else dfd.index.max())
    else:
        last_date = dfd.index.max()
    return ExpiryInfo(contract=contract, expiry_date=pd.Timestamp(last_date).normalize(), source="derived_from_data", source_url=None)


def merge_with_local_metadata(expiry_map: Dict[str, ExpiryInfo], meta: pd.DataFrame) -> Dict[str, ExpiryInfo]:
    """Override expiry info using a local CSV if present. Expected columns: root, contract, expiry_date, source, source_url."""
    if meta is None or meta.empty:
        return expiry_map
    meta = meta.copy()
    if "expiry_date" in meta.columns:
        meta["expiry_date"] = pd.to_datetime(meta["expiry_date"], errors="coerce")
    for _, row in meta.iterrows():
        contract = str(row.get("contract", ""))
        if not contract:
            continue
        ed = row.get("expiry_date")
        if pd.isna(ed):
            continue
        src = row.get("source", "local_metadata")
        src_url = row.get("source_url", None)
        expiry_map[contract] = ExpiryInfo(contract=contract, expiry_date=pd.Timestamp(ed).normalize(), source=str(src), source_url=(None if pd.isna(src_url) else str(src_url)))
    return expiry_map


def expiry_table_to_dataframe(expiry_map: Dict[str, ExpiryInfo]) -> pd.DataFrame:
    rows = []
    for c, info in sorted(expiry_map.items()):
        rows.append({
            "contract": info.contract,
            "expiry_date": pd.Timestamp(info.expiry_date).normalize(),
            "source": info.source,
            "source_url": info.source_url,
        })
    return pd.DataFrame(rows).sort_values(["expiry_date", "contract"]).reset_index(drop=True)
