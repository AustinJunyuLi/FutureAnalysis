"""
Unified command-line interface for futures roll analysis.

This module provides a single entry point for all CLI operations:
- analyze: Run hourly or daily analysis
- organize: Organize raw data files by commodity
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from . import analysis
from .config import load_settings

LOGGER = logging.getLogger(__name__)


def analyze_command(args: argparse.Namespace) -> int:
    """Execute analysis in hourly or daily mode."""
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s"
    )
    
    # Build overrides from command-line arguments
    overrides = {}
    if args.root:
        overrides.setdefault("data", {})["minute_root"] = args.root
    if args.metadata:
        overrides.setdefault("metadata", {})["contracts"] = args.metadata
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    
    if hasattr(args, "calendar") and args.calendar:
        overrides.setdefault("business_days", {})["calendar_paths"] = [args.calendar]
    
    # Load settings with overrides
    settings = load_settings(Path(args.settings), overrides=overrides or None)
    
    # Common arguments for both modes
    kwargs = {
        "settings": settings,
        "metadata_path": Path(args.metadata) if args.metadata else None,
        "output_dir": Path(args.output_dir) if args.output_dir else None,
    }
    
    # Run appropriate analysis
    if args.mode == "hourly":
        if hasattr(args, "max_files") and args.max_files:
            kwargs["max_files"] = args.max_files
        analysis.run_bucket_analysis(**kwargs)
        LOGGER.info("Hourly analysis finished.")
    elif args.mode == "daily":
        analysis.run_daily_analysis(**kwargs)
        LOGGER.info("Daily analysis finished.")
    else:
        LOGGER.error(f"Unknown mode: {args.mode}")
        return 1
    
    return 0


def organize_command(args: argparse.Namespace) -> int:
    """Execute data organization."""
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s"
    )
    
    source = Path(args.source).resolve()
    dest = Path(args.destination).resolve()
    inventory_path = Path(args.inventory).resolve()
    
    LOGGER.info("Scanning %s for raw futures files", source)
    txt_files = list(source.glob("*.txt"))
    inventory = []
    by_category: Dict[str, list[Tuple[Path, str, str]]] = defaultdict(list)
    
    # Import commodity map from organize module
    from .cli.organize import COMMODITY_MAP, _get_commodity_info
    
    for file_path in txt_files:
        symbol, folder, name = _get_commodity_info(file_path.name)
        by_category[folder].append((file_path, symbol or "UNKNOWN", name))
        size_mb = file_path.stat().st_size / (1024 * 1024)
        inventory.append({
            "filename": file_path.name,
            "symbol": symbol or "UNKNOWN",
            "commodity": name,
            "category": folder,
            "size_mb": f"{size_mb:.2f}",
        })
    
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
    
    pd.DataFrame(
        sorted(inventory, key=lambda x: (x["category"], x["symbol"], x["filename"]))
    ).to_csv(inventory_path, index=False)
    LOGGER.info("Organised %s files. Inventory written to %s", moved, inventory_path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for the unified CLI."""
    parser = argparse.ArgumentParser(
        prog="futures-roll",
        description="Futures Roll Analysis - Unified CLI for analyzing institutional roll patterns"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run futures roll analysis",
        description="Analyze futures data in hourly (bucket) or daily mode"
    )
    analyze_parser.add_argument(
        "--mode",
        choices=["hourly", "daily"],
        required=True,
        help="Analysis mode: hourly (bucket) or daily aggregation"
    )
    analyze_parser.add_argument(
        "--settings",
        default="config/settings.yaml",
        help="Path to the YAML settings file (default: config/settings.yaml)"
    )
    analyze_parser.add_argument(
        "--root",
        help="Override minute data root directory"
    )
    analyze_parser.add_argument(
        "--metadata",
        help="Override metadata CSV path"
    )
    analyze_parser.add_argument(
        "--output-dir",
        help="Directory for outputs"
    )
    analyze_parser.add_argument(
        "--max-files",
        type=int,
        help="Limit number of files for quick runs (hourly mode only)"
    )
    analyze_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    analyze_parser.add_argument(
        "--calendar",
        help="Path to trading calendar CSV file (overrides config). Use for custom holiday schedules."
    )
    
    # Organize command
    organize_parser = subparsers.add_parser(
        "organize",
        help="Organize raw futures data files",
        description="Organize raw futures files by commodity into structured directories"
    )
    organize_parser.add_argument(
        "--source",
        default=".",
        help="Directory containing raw futures files (default: current directory)"
    )
    organize_parser.add_argument(
        "--destination",
        default="organized_data",
        help="Destination directory for organized data (default: organized_data)"
    )
    organize_parser.add_argument(
        "--inventory",
        default="data_inventory.csv",
        help="Path for the generated inventory CSV (default: data_inventory.csv)"
    )
    organize_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    organize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned moves without performing them"
    )
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    # Execute appropriate command
    if args.command == "analyze":
        return analyze_command(args)
    elif args.command == "organize":
        return organize_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
