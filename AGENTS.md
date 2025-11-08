# Repository Guidelines

## Project Structure & Modules
- `src/futures_roll_analysis/` – core package (ingest, buckets, analysis, spreads, rolls, trading_days, reporting, results, `unified_cli.py`, `cli/`).
- `tests/` – pytest suites for buckets, rolls, trading_days, reporting, multi‑spread.
- `config/settings.yaml` – main configuration; `metadata/` holds calendars/contract metadata.
- `organized_data/` – raw minute data (large; excluded from git).
- `outputs/` – analysis artifacts and figures (excluded; `.gitkeep` allowed).
- `presentation_docs/` – LaTeX sources (`analytical_results_report.tex`, `technical_implementation_report.tex`, `Makefile`).

## Build, Test, and Run
- Create env: `conda create -n futures-roll python=3.11 && conda activate futures-roll`.
- Install: `pip install -e .[dev,viz]`.
- Hourly quick run (no report):
  `futures-roll analyze --mode hourly --settings config/settings.yaml --max-files 5 --skip-report`.
- Reports (from `presentation_docs/`):
  - `make assets` (generates panel/widening and figures)
  - `make results` (build analytical results PDF)
  - `make tech` (build technical implementation PDF)
- Direct report via CLI (default path):
  `futures-roll analyze --mode all --settings config/settings.yaml --report-path presentation_docs/technical_implementation_report.tex`.

## Coding Style & Conventions
- Python 3.11+, PEP 8, 4‑space indents; add type hints for new/changed code.
- Names: modules/functions `snake_case`; classes `CapWords`; constants `UPPER_SNAKE_CASE`.
- Use `pathlib` and `logging.getLogger(__name__)`; avoid `print` in library code.
- Do not hard‑code paths; read from `config/settings.yaml` or CLI overrides.

## Testing Guidelines
- Unit tests in `tests/test_<module>.py`; prefer Arrange–Act–Assert.
- Cover: calendar edge cases, DST, sparse/near‑expiry data, and error paths.
- Run: `pytest -q` • Coverage: `pytest --cov=src/futures_roll_analysis --cov-report=term-missing`.

## Commits & Pull Requests
- Conventional Commits: `feat|fix|chore|docs|refactor|test: short imperative summary`.
- PRs must include purpose, rationale, sample CLI, and (when relevant) before/after plots; link issues.

## Security & Configuration
- Keep `organized_data/` and generated `outputs/` out of version control (see `.gitignore`).
- Never commit secrets; use env vars (e.g., `FUTURES_PROJECT`) and relative paths.
- Trading calendars are required: set `business_days.calendar_paths` in `config/settings.yaml`.

