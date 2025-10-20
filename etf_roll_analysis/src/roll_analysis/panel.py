from typing import Dict

import numpy as np
import pandas as pd


def assemble_panel(daily_by_contract: Dict[str, pd.DataFrame], expiry_df: pd.DataFrame) -> pd.DataFrame:
    """Create a wide panel with columns (contract, field) and attach expiry info.
    expiry_df columns: [contract, expiry_date, source, source_url]
    """
    frames = []
    for contract, dfd in daily_by_contract.items():
        dfi = dfd.copy()
        # Attach expiry row (constant per contract)
        row = expiry_df.loc[expiry_df["contract"] == contract]
        exp = row["expiry_date"].iloc[0] if not row.empty else pd.NaT
        src = row["source"].iloc[0] if not row.empty else np.nan
        src_url = row["source_url"].iloc[0] if not row.empty else np.nan
        dfi["expiry"] = exp
        dfi["expiry_source"] = src
        dfi["expiry_source_url"] = src_url
        dfi.columns = pd.MultiIndex.from_product([[contract], dfi.columns])
        frames.append(dfi)

    panel = pd.concat(frames, axis=1).sort_index()
    return panel
