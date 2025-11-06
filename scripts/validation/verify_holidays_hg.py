#!/usr/bin/env python3
"""
Compare the repo's CME/Globex holiday calendar against an authoritative CSV
for COMEX Copper (HG) and report differences.

Usage:
  python scripts/verify_holidays_hg.py --authoritative /path/to/authoritative.csv \
      [--start 2015] [--end 2025]

Authoritative CSV format (flexible):
  - Must contain a 'date' column (YYYY-MM-DD).
  - Must contain a status indicator in either 'status' or 'session_note' or 'holiday_name'.
  - Values considered closed: status/holiday_name contains 'Closed' (case-insensitive) and not 'Early'.
  - Values considered early-close: session_note contains 'Early' (treated as trading day for our logic).

The script treats early-close days as trading days and flags only full 'Closed' days.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd


def _load_repo_calendar(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "metadata/calendars/cme_globex_holidays.csv"
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    for c in ('holiday_name','session_note','partial_hours'):
        if c not in df.columns:
            df[c] = ''
        df[c] = df[c].fillna('').astype(str)
    # Closed if holiday and not early-close
    closed = (df['holiday_name'].str.strip().str.len() > 0) & (
        df['session_note'].str.strip().str.lower() != 'early close'
    )
    return pd.DataFrame({'date': df['date'], 'closed': closed}).dropna(subset=['date']).drop_duplicates('date')


def _load_authoritative(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'date' not in df.columns:
        raise ValueError("Authoritative CSV must include a 'date' column (YYYY-MM-DD)")
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    # Try to derive closed status
    # Priority: explicit 'closed' bool, then 'status', then 'session_note', then 'holiday_name'
    closed = None
    if 'closed' in df.columns:
        closed = df['closed'].astype(bool)
    else:
        candidates = ['status', 'session_note', 'holiday_name']
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        if found is None:
            raise ValueError("Authoritative CSV must include one of: closed, status, session_note, holiday_name")
        series = df[found].fillna('').astype(str)
        # Closed if contains 'Closed' and not 'Early'
        s_low = series.str.lower()
        closed = series.str.contains('Closed', case=False) & (~s_low.str.contains('early'))
    return pd.DataFrame({'date': df['date'], 'closed': closed}).dropna(subset=['date']).drop_duplicates('date')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify CME/Globex holiday calendar for COMEX HG")
    ap.add_argument('--authoritative', required=True, help='Path to authoritative CSV')
    ap.add_argument('--start', type=int, default=2015, help='Start year (inclusive)')
    ap.add_argument('--end', type=int, default=2025, help='End year (inclusive)')
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    repo_cal = _load_repo_calendar(repo_root)
    auth_cal = _load_authoritative(Path(args.authoritative))

    # Restrict to year range
    repo_cal['year'] = repo_cal['date'].dt.year
    auth_cal['year'] = auth_cal['date'].dt.year
    repo_cal = repo_cal[(repo_cal['year'] >= args.start) & (repo_cal['year'] <= args.end)]
    auth_cal = auth_cal[(auth_cal['year'] >= args.start) & (auth_cal['year'] <= args.end)]

    # Build sets of closed dates
    repo_closed = set(repo_cal.loc[repo_cal['closed'], 'date'])
    auth_closed = set(auth_cal.loc[auth_cal['closed'], 'date'])

    missing_in_repo = sorted(auth_closed - repo_closed)
    extra_in_repo = sorted(repo_closed - auth_closed)

    print(f"Checked years {args.start}-{args.end}")
    print(f"Authoritative closed days: {len(auth_closed)} | Repo closed days: {len(repo_closed)}")
    print()
    if missing_in_repo:
        print("Missing (closed in authoritative, not closed in repo):")
        for d in missing_in_repo:
            print(f"  - {d.date()}")
    else:
        print("No missing closed days.")
    print()
    if extra_in_repo:
        print("Extra (closed in repo, not closed in authoritative):")
        for d in extra_in_repo:
            print(f"  - {d.date()}")
    else:
        print("No extra closed days.")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

