# dnr-synth

`dnr-synth` is a Typer-based CLI that fabricates messy-yet-controlled analytics datasets, dbt skeletons, and evaluation artifacts for three domains: fintech, ecommerce, and SaaS. It helps stress-test a Data Needs Reporter by delivering deterministic generators, corruptors, and health reports.

## Features
- Deterministic synthetic frames powered by NumPy, pandas, pyarrow, and Faker.
- Configurable corruptors for key presence, joinability, lateness, duplicates, null spikes, and schema drift.
- Turn-key dbt starter projects plus health metrics (JSON + Markdown) for quick validation.
- Lightweight sampling of NL queries and Slack threads for UX demos.

## Quickstart
```bash
poetry install

# 1) Scaffold domain assets
poetry run dnr-synth init fintech

# 2) Generate messy parquet outputs
poetry run dnr-synth generate --config domains/fintech/config.yaml --seed 4242

# 3) Evaluate data health and review artifacts
poetry run dnr-synth evaluate --domain fintech
```

## Commands
| Command | Description |
| --- | --- |
| `dnr-synth init <domain>` | Scaffold `domains/<domain>/config.yaml`, `dbt_<domain>/`, and `artifacts/<domain>/`. |
| `dnr-synth generate --config <yaml> [--out <dir>] [--seed <int>] [--dry-run]` | Produce clean data, apply corruptors, and write Parquet partitioned by `ds`. |
| `dnr-synth evaluate --domain <domain> [--data <dir>] [--out <dir>]` | Compute health metrics (joinability, key nulls, lag, duplicates, null spikes). |
| `dnr-synth sample --domain <domain> --data <dir> --dbt <dir> [--out <dir>] [--seed <int>]` | Emit grounded `nl_queries.json` and `slack_threads.json` derived from the generated datasets. |
| `dnr-synth preview <folder> [--limit N]` | Pretty-print up to N rows per dataset plus sample NL queries and Slack threads. |
| `dnr-synth export --to postgres --dsn <url>` | Placeholder stub for future warehouse exports. |

## Configuration
- Configs live under `domains/<domain>/config.yaml` (Pydantic-validated, loaded via ruamel.yaml).
- Randomness uses `numpy.random.Generator` with `PCG64`; omit `--seed` to default to `7`.
- Parquet outputs land under `data/<domain>` by default (`parquet://` URIs resolve to local paths).

## Development
- Python 3.11+, managed via Poetry (`poetry install`, `poetry run pytest`).
- Templates ship under `templates/` and are copied with token substitution on `init`.
- Tests cover corruptor probabilities and evaluator metrics to keep regressions in check.

## License
MIT
