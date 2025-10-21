# Futures Roll Analysis – Agent Anchor

## Overview
This repository provides a Python package, `futures_roll_analysis`, for loading
minute-level futures data, aggregating it to daily or session buckets, and
detecting institutional roll activity through calendar-spread analytics. The
codebase was restructured into a conventional `src/` layout with shared service
layers for ingest, panel assembly, roll identification, event detection, and
CLI entry points.

## Layout
```
config/                 # Project configuration (YAML)
metadata/               # Official contract metadata (CSV)
organized_data/         # Raw minute data organised by commodity (external)
outputs/                # Generated panels, signals, summaries
presentation_docs/      # LaTeX sources and published report
src/futures_roll_analysis/
  ├── analysis.py       # High-level bucket & daily analysis runners
  ├── buckets.py        # Bucket definitions and aggregation routines
  ├── cli/              # Command-line entry points (hourly, daily, organise)
  ├── config.py         # Settings loader with override support
  ├── events.py         # Spread event detection & summaries
  ├── ingest.py         # Minute data ingestion and contract aggregation
  ├── panel.py          # Panel assembly from per-contract frames
  ├── quality.py        # Data-quality filtering & reporting helpers
  └── rolls.py          # Vectorised front/next, spread & liquidity utilities
src/tests/              # Pytest suites for buckets/events
```

## Installation
```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev,viz]
```
The editable install exposes console entry points registered in
`pyproject.toml`.

## Preparing Data
Raw text files can be organised into `organized_data/` using:
```bash
python -m futures_roll_analysis.cli.organize --source <raw_dir> --destination organized_data
```
An inventory CSV is written alongside the organised folders.

## Running Analyses
Hourly (bucket) pipeline:
```bash
python -m futures_roll_analysis.cli.hourly \
  --settings config/settings.yaml \
  --root organized_data/copper \
  --output-dir outputs
```
Daily pipeline with data-quality filtering:
```bash
python -m futures_roll_analysis.cli.daily \
  --settings config/settings.yaml \
  --root organized_data/copper \
  --output-dir outputs
```
Both commands honour optional `--metadata`, `--max-files`, and `--log-level`
overrides.

## Outputs
- `outputs/panels/` – Parquet and CSV panels with multi-contract fields.
- `outputs/roll_signals/` – Spread/widening/liquidity series.
- `outputs/analysis/` – Bucket summaries, preference scores, transition matrices,
  and daily widening summaries.
- `outputs/data_quality/` – CSV/JSON quality metrics when daily filtering runs.

## Testing & Quality
Run the automated suites with:
```bash
pytest
```
The refactor introduces vectorised front/next contract detection, shared
spread-event logic, and centralised configuration loading to remove script
duplication and `sys.path` hacks while maintaining previous analytical results.
