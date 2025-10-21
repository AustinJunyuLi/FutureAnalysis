from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def assemble_panel(
    frames_by_contract: Dict[str, pd.DataFrame],
    expiry_metadata: pd.DataFrame,
    *,
    include_bucket_meta: bool = True,
) -> pd.DataFrame:
    """
    Assemble a wide panel with MultiIndex columns (contract, field).

    Parameters
    ----------
    frames_by_contract:
        Dictionary mapping contract codes to aggregated DataFrames.
    expiry_metadata:
        DataFrame with at least ``contract`` and ``expiry_date`` columns.
    include_bucket_meta:
        When ``True`` (default) bucket metadata columns are projected into a ``meta`` namespace.
    """
    if not frames_by_contract:
        raise ValueError("No contract frames supplied to assemble_panel")

    meta_index = expiry_metadata.drop_duplicates("contract").set_index("contract")
    frames: List[pd.DataFrame] = []
    bucket_meta_cols: Optional[List[str]] = None
    bucket_meta_frames: List[pd.DataFrame] = []

    for contract, frame in sorted(frames_by_contract.items()):
        df = frame.copy()

        if contract not in meta_index.index:
            raise ValueError(f"Missing expiry metadata for contract {contract}")

        meta_row = meta_index.loc[contract]
        df["expiry"] = pd.to_datetime(meta_row["expiry_date"]).normalize()
        df["expiry_source"] = meta_row.get("source", np.nan)
        df["expiry_source_url"] = meta_row.get("source_url", np.nan)

        if bucket_meta_cols is None:
            bucket_meta_cols = [
                col
                for col in ["bucket", "bucket_label", "session"]
                if col in df.columns
            ]

        if bucket_meta_cols:
            bucket_meta_frames.append(df[bucket_meta_cols])
            df = df.drop(columns=bucket_meta_cols)

        columns = pd.MultiIndex.from_product([[contract], df.columns])
        df.columns = columns
        frames.append(df)

    panel = pd.concat(frames, axis=1).sort_index()

    if include_bucket_meta and bucket_meta_cols:
        bucket_meta = _merge_bucket_meta(bucket_meta_frames, panel.index, bucket_meta_cols)
        for col in bucket_meta_cols:
            panel[("meta", col)] = bucket_meta[col]

    return panel


def _merge_bucket_meta(
    frames: List[pd.DataFrame], index: pd.Index, columns: List[str]
) -> pd.DataFrame:
    meta = pd.concat(frames, axis=0)
    meta = meta.loc[:, columns]
    meta = meta[~meta.index.duplicated(keep="first")]
    meta = meta.reindex(index)
    return meta
