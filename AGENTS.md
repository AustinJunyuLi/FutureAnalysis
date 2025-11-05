# Repository Guidelines

## Project Structure & Module Organization
- `src/futures_roll_analysis/` – core package (ingest, buckets, analysis, spreads, rolls, trading_days, calendar_tools, `unified_cli.py`, `cli/{hourly,daily,organize}.py`).
- `tests/` – pytest suite (`test_*.py`).
- `config/settings.yaml` – default runtime configuration.
- `organized_data/` – minute data inputs (large; keep out of git).
- `outputs/` – generated panels, roll signals, summaries.
- `scripts/` – helpers (e.g., `setup_env.sh`, `cleanup.sh`).
- `presentation_docs/` – LaTeX report (`Makefile`, figures, tables).

## Build, Test, and Development Commands
- Create env (conda recommended): `conda create -n futures-roll python=3.11 && conda activate futures-roll`.
- Install for dev: `pip install -e .[dev,viz]`.
- Run CLI (hourly sample): `futures-roll analyze --mode hourly --settings config/settings.yaml --max-files 5 --output-dir outputs/test`.
- Organize raw files (dry run): `futures-roll organize --source raw_data --destination organized_data --dry-run`.
- Run tests: `pytest -q` • With coverage: `pytest --cov=src/futures_roll_analysis --cov-report=term-missing`.
- Build/view report (from `presentation_docs/`): `make analysis` • `make view-analysis` • cleanup: `make clean`.

## Coding Style & Naming Conventions
- Python 3.11+, PEP 8, 4‑space indents; add type hints for new/changed functions.
- Names: modules/functions `snake_case`; classes `CapWords`; constants `UPPER_SNAKE_CASE`.
- Use `pathlib` and `logging.getLogger(__name__)`; avoid `print` in library code.
- Configuration comes from `config/settings.yaml` (or CLI overrides); do not hard‑code paths.

## Testing Guidelines
- Put tests under `tests/test_<module>.py`; follow Arrange–Act–Assert.
- Cover edge calendars/holidays, sparse or near‑expiry data, and error paths.
- Maintain or raise coverage for touched files (target ≥80% for changes).

## Commit & Pull Request Guidelines
- Prefer Conventional Commits: `feat|fix|chore|docs|refactor|test: short imperative summary` (e.g., `chore: tighten ignore rules for data/outputs`).
- PRs: include purpose, rationale, sample CLI invocation, and links to issues; attach before/after snippets or plots when relevant.

## Security & Configuration Tips
- Keep large data in `organized_data/` and generated `outputs/` out of version control; confirm `.gitignore` before committing.
- Never commit credentials; use env vars (e.g., `FUTURES_PROJECT`) and relative paths.

## Agent‑Specific Instructions
- Make minimal, focused patches; keep public CLI names stable (`futures-roll`).
- If changing settings schema, update `config/settings.yaml`, README, and tests accordingly.
