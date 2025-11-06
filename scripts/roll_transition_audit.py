#!/usr/bin/env python3
from __future__ import annotations

"""
Roll Transition Audit

Build an audit of observed F1 transitions from the panel and compare each
transition start time to the previous F1's expiry timestamp.

Outputs: outputs/analysis/roll_transition_audit.csv
"""

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


def build_f1_transition_chain(index: pd.DatetimeIndex, f1_series: pd.Series) -> pd.DataFrame:
    """Return observed F1 transition segments with start/end positions."""
    idx = pd.DatetimeIndex(index)
    s = pd.Series(f1_series.values, index=idx).astype(object)
    changes = s.ne(s.shift(1)).fillna(True)
    change_points = s.index[changes]
    starts = list(change_points)
    positions = np.searchsorted(idx.values, np.array(starts, dtype="datetime64[ns]"))
    segments: List[Dict] = []
    for i, (t0, pos0) in enumerate(zip(starts, positions)):
        pos1 = positions[i + 1] if i + 1 < len(positions) else len(idx)
        new_f1 = s.loc[t0]
        prev_f1 = s.iloc[pos0 - 1] if pos0 > 0 else None
        segments.append(
            {
                "start_time": t0,
                "start_pos": int(pos0),
                "end_pos": int(pos1),
                "new_f1": new_f1,
                "prev_f1": prev_f1,
            }
        )
    return pd.DataFrame(segments)


def main() -> int:
    base = Path(__file__).resolve().parents[1]
    panel_path = base / "outputs" / "panels" / "hourly_panel.parquet"
    metadata_path = base / "metadata" / "contracts_metadata.csv"
    out_dir = base / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "roll_transition_audit.csv"

    print("Loading panel:", panel_path)
    panel = pd.read_parquet(panel_path)
    f1 = panel["meta"]["front_contract"]

    print("Loading metadata:", metadata_path)
    metadata = pd.read_csv(metadata_path)
    metadata["expiry_date"] = pd.to_datetime(metadata["expiry_date"]).dt.normalize()
    # Default expiry time at 17:00 local (naive), consistent with rolls.build_expiry_map
    metadata["expiry_ts"] = metadata["expiry_date"] + pd.to_timedelta(17, unit="h")
    expiry_map = metadata.set_index("contract")["expiry_ts"]

    print("Building observed transition chain…")
    chain = build_f1_transition_chain(panel.index, f1)

    print("Computing audit rows…")
    rows: List[Dict] = []
    idx = pd.DatetimeIndex(panel.index)
    for _, seg in chain.iterrows():
        prev = seg["prev_f1"]
        if prev is None or (isinstance(prev, float) and pd.isna(prev)):
            # No previous F1 for the first segment
            continue
        start_pos = int(seg["start_pos"])
        start_time = seg["start_time"]
        new_f1 = seg["new_f1"]
        last_prev_pos = start_pos - 1
        last_prev_time = idx[last_prev_pos] if last_prev_pos >= 0 else pd.NaT

        prev_expiry = expiry_map.get(str(prev), pd.NaT)
        if pd.isna(prev_expiry):
            delta = pd.NaT
            delta_seconds = np.nan
        else:
            # Naive timestamps: panel index is naive local; expiry_ts is naive local
            delta = pd.Timestamp(start_time) - pd.Timestamp(prev_expiry)
            delta_seconds = delta.total_seconds()

        rows.append(
            {
                "prev_f1": prev,
                "new_f1": new_f1,
                "prev_f1_expiry_ts": prev_expiry,
                "last_prev_f1_time": last_prev_time,
                "transition_start_time": start_time,
                "delta_seconds": delta_seconds,
                "delta_hours": (delta_seconds / 3600.0) if pd.notna(delta_seconds) else np.nan,
                "delta_days": (delta_seconds / 86400.0) if pd.notna(delta_seconds) else np.nan,
            }
        )

    audit = pd.DataFrame(rows)
    audit.sort_values("transition_start_time", inplace=True)
    audit.to_csv(out_path, index=False)
    print(f"Audit written to: {out_path}")

    if not audit.empty:
        neg = int((audit["delta_seconds"] < 0).sum())
        zero = int((audit["delta_seconds"] == 0).sum())
        pos = int((audit["delta_seconds"] > 0).sum())
        print(f"Summary: {len(audit)} transitions | before-expiry: {neg} | at-expiry: {zero} | after-expiry: {pos}")
        print("Delta (hours): min=%.2f median=%.2f max=%.2f" % (
            np.nanmin(audit["delta_hours"].values),
            np.nanmedian(audit["delta_hours"].values),
            np.nanmax(audit["delta_hours"].values),
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

