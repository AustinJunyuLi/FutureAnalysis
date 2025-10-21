from __future__ import annotations

import argparse
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

COMMODITY_MAP: Dict[str, Tuple[str, str]] = {
    "HG": ("copper", "Copper"),
    "GC": ("gold", "Gold"),
    "SI": ("silver", "Silver"),
    "PL": ("platinum", "Platinum"),
    "PA": ("palladium", "Palladium"),
    "ALI": ("aluminum", "Aluminum"),
    "CL": ("crude_oil", "Crude Oil WTI"),
    "BZ": ("brent_crude", "Brent Crude"),
    "NG": ("natural_gas", "Natural Gas"),
    "HO": ("heating_oil", "Heating Oil"),
    "RB": ("gasoline", "RBOB Gasoline"),
    "ZC": ("corn", "Corn"),
    "ZW": ("wheat", "Wheat"),
    "ZS": ("soybeans", "Soybeans"),
    "ZL": ("soybean_oil", "Soybean Oil"),
    "ZM": ("soybean_meal", "Soybean Meal"),
    "ZO": ("oats", "Oats"),
    "ZR": ("rice", "Rice"),
    "SB": ("sugar", "Sugar"),
    "KC": ("coffee", "Coffee"),
    "CC": ("cocoa", "Cocoa"),
    "CT": ("cotton", "Cotton"),
    "OJ": ("orange_juice", "Orange Juice"),
    "LE": ("live_cattle", "Live Cattle"),
    "HE": ("lean_hogs", "Lean Hogs"),
    "GF": ("feeder_cattle", "Feeder Cattle"),
    "AD": ("currencies", "Australian Dollar"),
    "A6": ("currencies", "Australian Dollar (micro)"),
    "DX": ("currencies", "US Dollar Index"),
    "CNH": ("currencies", "Offshore RMB"),
    "JY": ("currencies", "Japanese Yen"),
    "PJY": ("currencies", "Japanese Yen (mini)"),
    "NOK": ("currencies", "Norwegian Krone"),
    "SEK": ("currencies", "Swedish Krona"),
    "KRW": ("currencies", "Korean Won"),
    "PRK": ("currencies", "Polish Zloty"),
    "TWN": ("currencies", "Taiwan Dollar"),
    "ES": ("equity_indices", "E-mini S&P 500"),
    "MES": ("equity_indices", "Micro E-mini S&P 500"),
    "NQ": ("equity_indices", "E-mini NASDAQ-100"),
    "MNQ": ("equity_indices", "Micro E-mini NASDAQ-100"),
    "YM": ("equity_indices", "E-mini Dow"),
    "RTY": ("equity_indices", "E-mini Russell 2000"),
    "NK": ("equity_indices", "Nikkei 225 (USD)"),
    "NKD": ("equity_indices", "Nikkei 225 (Yen)"),
    "NIY": ("equity_indices", "Nikkei 225 Yen"),
    "ZN": ("fixed_income", "10-Year T-Note"),
    "ZB": ("fixed_income", "30-Year T-Bond"),
    "ZF": ("fixed_income", "5-Year T-Note"),
    "ZT": ("fixed_income", "2-Year T-Note"),
    "ZQ": ("fixed_income", "30-Day Fed Funds"),
    "UB": ("fixed_income", "Ultra T-Bond"),
    "TN": ("fixed_income", "Ultra 10-Year"),
    "BTC": ("crypto", "Bitcoin"),
    "MBT": ("crypto", "Micro Bitcoin"),
    "ETH": ("crypto", "Ethereum"),
    "MET": ("crypto", "Micro Ether"),
    "VX": ("volatility", "VIX Futures"),
    "VXM": ("volatility", "Mini VIX"),
}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Organise raw futures data files by commodity.")
    parser.add_argument(
        "--source",
        default=".",
        help="Directory containing raw futures files.",
    )
    parser.add_argument(
        "--destination",
        default="organized_data",
        help="Destination directory for organised data.",
    )
    parser.add_argument(
        "--inventory",
        default="data_inventory.csv",
        help="Path of the generated inventory CSV.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned moves without performing them.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    source = Path(args.source).resolve()
    dest = Path(args.destination).resolve()
    inventory_path = Path(args.inventory).resolve()

    LOGGER.info("Scanning %s for raw futures files", source)
    txt_files = list(source.glob("*.txt"))
    inventory = []
    by_category: Dict[str, list[Tuple[Path, str, str]]] = defaultdict(list)

    for file_path in txt_files:
        symbol, folder, name = _get_commodity_info(file_path.name)
        by_category[folder].append((file_path, symbol or "UNKNOWN", name))
        size_mb = file_path.stat().st_size / (1024 * 1024)
        inventory.append(
            {
                "filename": file_path.name,
                "symbol": symbol or "UNKNOWN",
                "commodity": name,
                "category": folder,
                "size_mb": f"{size_mb:.2f}",
            }
        )

    if args.dry_run:
        for folder, files in sorted(by_category.items()):
            LOGGER.info("%s/: %s files (dry run)", folder, len(files))
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    for folder, files in sorted(by_category.items()):
        target_dir = dest / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        for src_file, _, _ in files:
            target = target_dir / src_file.name
            if not target.exists():
                shutil.move(str(src_file), str(target))
                moved += 1
        LOGGER.info("%s/: %s files organised", folder, len(files))

    pd.DataFrame(sorted(inventory, key=lambda x: (x["category"], x["symbol"], x["filename"]))).to_csv(
        inventory_path, index=False
    )
    LOGGER.info("Organised %s files. Inventory written to %s", moved, inventory_path)
    return 0


def _get_commodity_info(filename: str) -> Tuple[Optional[str], str, str]:
    parts = filename.split("_")
    if len(parts) >= 2:
        symbol = parts[0]
        if symbol in COMMODITY_MAP:
            folder, name = COMMODITY_MAP[symbol]
            return symbol, folder, name
    return None, "other", "Unknown"


if __name__ == "__main__":
    raise SystemExit(main())
