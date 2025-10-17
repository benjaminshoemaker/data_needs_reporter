# Repository Guidelines

## Project Structure & Module Organization
- `dnr_synth/` — main package
  - `domains/` clean generators (fintech, ecom, saas)
  - `corruptors/` keys, joinability, lateness, duplicates, null spikes, schema drift
  - `writers/` Parquet writer
  - `eval/` health metrics + markdown report
  - `sample/` grounded NL/Slack generator, validators, providers
  - `cli.py` Typer entrypoint (`dnr-synth`)
- `templates/` dbt skeleton + config examples
- `tests/` pytest suite
- Generated (git‑ignored): `data/`, `artifacts/`

## Build, Test, and Development Commands
- Install: `poetry install`
- Run tests: `poetry run pytest -q`
  - CI runs tests on Python 3.11 and 3.12.
  - CI treats `DeprecationWarning`/`PendingDeprecationWarning` as errors, with a targeted ignore for a pandas internal warning related to BlockManager construction during PyArrow -> pandas conversion.
- Generate demo data (fintech):
  - `poetry run dnr-synth init fintech`
  - `poetry run dnr-synth generate --config domains/fintech/config.yaml --seed 4242`
  - `poetry run dnr-synth evaluate --domain fintech`
- Grounded samples (no network):
  - `poetry run dnr-synth sample --domain fintech --data data/fintech --dbt dbt_fintech --seed 4242`
- Manual prompts (use ChatGPT, save JSON back):
  - `poetry run dnr-synth prompts --domain fintech --data data/fintech --dbt dbt_fintech`
  - Save outputs to `artifacts/fintech/nl_queries.json` and `artifacts/fintech/slack_threads.json`
- Preview: `poetry run dnr-synth preview fintech`

### Gaps Report (LLM prompt & optional run)
- Build an LLM-ready gaps analysis prompt from generated artifacts:
  - `poetry run dnr-synth gaps-report --domain fintech`
  - Defaults: data `data/<domain>`, artifacts `artifacts/<domain>`, writes to `artifacts/<domain>/gaps/PROMPT_GAPS.md`.
- Optional: call an LLM via OpenAI Responses API (requires env key):
  - `poetry run dnr-synth gaps-report --domain fintech --run`
  - API key resolution: `LLM_API_KEY` or `OPENAI_API_KEY`; override API base via `--api-base`.
  - Writes `report.md`, `report_text.md` (raw markdown from the LLM payload), and `data_gaps_report.json` when provided; otherwise saves raw output to `llm_output.txt`.
  - Increase request timeout for slower models via `--timeout-s` (default 120s).

### Legacy Gap Report
- Removed. The legacy datagap_report tool and example fixtures were deleted to simplify the repo and focus on data generation and grounded samples.

## Coding Style & Naming Conventions
- Python 3.11+ (CI: 3.11, 3.12), PEP 8, 4‑space indent, ~100 char lines.
- Use type hints; Pydantic models for configs/context.
- Prefer Pydantic v2 validators (e.g., `@field_validator`), not deprecated v1 `@validator`.
- Names: files/modules `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`.
- Keep CLI flags stable; preserve Parquet/artifact formats.
 - Pandas: use lowercase frequency strings (e.g., `"h"` not `"H"`). Avoid patterns that trigger dtype downcasting warnings (pre-seed columns before concat/fillna where needed).

## Testing Guidelines
- Framework: pytest; files `tests/test_*.py`.
- Prefer deterministic seeds; synth test data should be small and in‑memory or temp Parquet.
- Aim for focused unit tests near changed logic (corruptors, evaluators, samplers).
 - Keep tests warning-clean. If a third-party deprecation is unavoidable, add a narrow `pytest.ini` filter for that specific message/module.

## CI
- GitHub Actions workflow lives at `.github/workflows/ci.yml`.
- Uses Poetry to install deps and caches `.venv` keyed by `poetry.lock`.
- Executes `pytest` with deprecations-as-errors; see `PYTHONWARNINGS` in the workflow for the specific ignore in place for pandas BlockManager deprecation during PyArrow conversion.

## Commit & Pull Request Guidelines
- Commits: imperative, scoped (e.g., `sample: diversify slack scenarios`).
- Don’t commit generated `data/` or `artifacts/`.
- PRs include: description, repro steps/commands, sample paths (e.g., `artifacts/fintech/health_profile.json`), and linked issues.

## Security & Configuration Tips
- Project runs offline by default. Optional LLM provider is stubbed; don’t hardcode secrets.
- Use environment variables only when wiring custom providers locally.
