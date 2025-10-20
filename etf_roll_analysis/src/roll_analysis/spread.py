import pandas as pd


def compute_spread(panel: pd.DataFrame, price_field: str = "close") -> pd.Series:
    f = panel[("meta", "front_contract")].astype("string")
    n = panel[("meta", "next_contract")].astype("string")
    vals = []
    for dt in panel.index:
        fc = f.loc[dt]
        nc = n.loc[dt]
        try:
            pf = panel.at[dt, (fc, price_field)] if pd.notna(fc) else float("nan")
            pn = panel.at[dt, (nc, price_field)] if pd.notna(nc) else float("nan")
            vals.append(pn - pf)
        except Exception:
            vals.append(float("nan"))
    return pd.Series(vals, index=panel.index, name=f"spread_{price_field}_next_minus_front")
