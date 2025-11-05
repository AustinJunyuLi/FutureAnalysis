from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)


ALLOWED_SESSION_NOTES = {
    "": "unspecified",
    "regular": "open",
    "early close": "partial",
    "closed": "closed",
    "full close": "closed",
}


@dataclass(frozen=True)
class CalendarIssue:
    level: str  # "error" | "warning"
    message: str
    date: Optional[str] = None


def _load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    required = {"date", "holiday_name", "session_note", "partial_hours"}
    missing = required - set(df.columns)
    for m in missing:
        df[m] = ""
    df = df[list(required)]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    for c in ("holiday_name", "session_note", "partial_hours"):
        df[c] = df[c].fillna("").astype(str)
    return df


def validate_calendar_df(df: pd.DataFrame) -> Tuple[List[CalendarIssue], pd.DataFrame]:
    issues: List[CalendarIssue] = []
    # Normalize
    work = df.copy()
    work["session_note_norm"] = work["session_note"].str.strip().str.lower()
    # Unknown session notes
    unknown_mask = ~work["session_note_norm"].isin(ALLOWED_SESSION_NOTES.keys())
    for _, row in work.loc[unknown_mask, :].iterrows():
        issues.append(CalendarIssue(
            level="error",
            message=f"Unknown session_note value: '{row['session_note']}'",
            date=str(row["date"].date()),
        ))
    # Duplicate dates
    dupes = work[work["date"].duplicated()]["date"].dt.date.astype(str).tolist()
    for d in dupes:
        issues.append(CalendarIssue(level="warning", message="Duplicate date", date=d))
    # Sort
    work = work.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    # is_trading_day classification (closed only if explicitly closed)
    closed = work["session_note_norm"].isin({"closed", "full close"})
    work["is_trading_day"] = ~closed
    return issues, work[["date", "holiday_name", "session_note", "partial_hours", "is_trading_day"]]


def lint(paths: Sequence[Path]) -> Tuple[int, List[CalendarIssue]]:
    code = 0
    all_issues: List[CalendarIssue] = []
    for path in paths:
        if not path.exists():
            all_issues.append(CalendarIssue(level="error", message=f"File not found: {path}"))
            code = 2
            continue
        try:
            df = _load_one(path)
            issues, _ = validate_calendar_df(df)
            all_issues.extend(issues)
            if any(i.level == "error" for i in issues):
                code = 1
        except Exception as exc:  # pragma: no cover
            all_issues.append(CalendarIssue(level="error", message=f"Failed to read {path}: {exc}"))
            code = 2
    return code, all_issues


def lint_main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Lint and validate trading calendar CSV files")
    parser.add_argument("paths", nargs="+", help="One or more calendar CSV paths")
    parser.add_argument("--json", dest="json_out", help="Optional JSON report path")
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.paths]
    code, issues = lint(paths)
    for i in issues:
        prefix = "ERROR" if i.level == "error" else "WARN"
        suffix = f" on {i.date}" if i.date else ""
        print(f"{prefix}: {i.message}{suffix}")

    if args.json_out:
        payload = [i.__dict__ for i in issues]
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON report to {args.json_out}")

    if code == 0:
        print("Calendar lint passed: no issues.")
    elif code == 1:
        print("Calendar lint completed with errors.")
    else:
        print("Calendar lint failed.")
    return code

