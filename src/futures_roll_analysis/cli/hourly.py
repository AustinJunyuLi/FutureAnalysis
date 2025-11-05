from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from .. import analysis
from ..config import load_settings


LOGGER = logging.getLogger(__name__)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run hourly (bucket) futures roll analysis.")
    parser.add_argument(
        "--settings",
        default="config/settings.yaml",
        help="Path to the YAML settings file.",
    )
    parser.add_argument("--root", help="Override minute data root directory.")
    parser.add_argument("--metadata", help="Override metadata CSV path.")
    parser.add_argument("--output-dir", help="Directory for outputs.")
    parser.add_argument("--max-files", type=int, help="Limit number of files for quick runs.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    overrides = {}
    if args.root:
        overrides.setdefault("data", {})["minute_root"] = args.root
    if args.metadata:
        overrides.setdefault("metadata", {})["contracts"] = args.metadata
    if args.output_dir:
        overrides["output_dir"] = args.output_dir

    settings = load_settings(Path(args.settings), overrides=overrides or None)

    analysis.run_bucket_analysis(
        settings,
        max_files=args.max_files,
        metadata_path=Path(args.metadata) if args.metadata else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    LOGGER.info("Hourly analysis finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
