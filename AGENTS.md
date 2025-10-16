# Repository Guidelines

## Project Structure & Module Organization
- `dnr_synth/`: Typer CLI (`dnr-synth`) with domain generators, corruptors, evaluators, and parquet writers.
- `templates/`: dbt skeleton and config examples copied during `dnr-synth init`.
- `domains/`: Per-domain configs created at runtime (git-ignored by default).
- `artifacts/`, `data/`: Generated health reports and parquet datasets (not committed).
- `tests/`: Pytest suite for corruptors and evaluators.
- Legacy `datagap_report/` remains for report tooling; no synthetic generation logic lives there.
- Examples/live outputs: created via `make fixtures`.

## Build, Test, and Development Commands
- Python 3.11 managed via Poetry: `poetry install`.
- Run tests: `poetry run pytest`.
- Generate sample data: `poetry run dnr-synth init fintech && poetry run dnr-synth generate --config domains/fintech/config.yaml`.
- Evaluate data health: `poetry run dnr-synth evaluate --domain fintech`.
- Sample queries/chat: `poetry run dnr-synth sample --domain fintech --data data/fintech --dbt dbt_fintech`.
- Preview datasets and artifacts: `poetry run dnr-synth preview fintech`.
- Fixture pipeline (data + artifacts): `make fixtures`.

## Coding Style & Naming Conventions
- Follow PEP 8; 4-space indent; max line length ~100.
- Use type hints and `from __future__ import annotations` where present.
- Naming: modules/files `snake_case.py`; functions/variables `snake_case`; classes `PascalCase`.
- Keep the `dnr-synth` CLI arguments stable; avoid breaking schema expectations for parquet outputs.

## Testing Guidelines
- Framework: pytest. Place tests under `tests/` named `test_*.py`.
- Prefer deterministic inputs; when generating data, pass `--seed` and commit only small fixtures under `fixtures/`.
- Run `poetry run pytest`; add targeted tests near new corruptors/evaluator logic.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope (e.g., `report: improve gap scoring`).
- Do not commit generated `data/` or `artifacts/` directories. Pre-commit should block these.
- PRs must include clear description, reproduction steps, and sample artifact paths (e.g., `artifacts/fintech_golden/health_profile.json`). Link related issues.

## Security & Configuration Tips
- LLM access: set `LLM_API_KEY` if using `datagap-report` flows (optional).
- Never commit secrets. Validate outputs (`dnr-synth evaluate --domain <domain>`) before sharing artifacts.

## Agent-Specific Instructions
- Keep changes scoped; avoid breaking public CLI flags.
- Preserve parquet and artifact formats expected by downstream tooling.
- Do not add or commit large generated data; use `make fixtures` to refresh examples.
