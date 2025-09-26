datagap_synth
=================

Minimal Python 3.11 utility to generate synthetic “telemetry packs” for data-question workflows, used to develop and validate data-gap detection/reporting tools.

Features
- CLI: `datagap-synth` with `init`, `gen`, `validate`, `list`.
- Generates packs with catalog, nl queries, Slack, and Email events.
- Deterministic when `--seed` is provided (including a deterministic clock).
- JSON Schema validation + cross-file checks.
- Standard library + PyYAML + jsonschema only.

Quickstart
1) Install in editable mode

    pip install -e .

2) Initialize config and schemas in your working directory

    datagap-synth init

3) Generate a pack

    datagap-synth gen --config config.yaml --out packs/vYYYY-MM-DD-a

4) Validate a pack

    datagap-synth validate --pack packs/vYYYY-MM-DD-a

5) List packs

    datagap-synth list --root packs

Data Gaps Report (LLM-assisted)
- CLI: `datagap-report`
- Uses OpenAI Responses API with Structured Outputs (JSON Schema) and optional embeddings to improve intent extraction, gap classification, clustering, and summaries. Caching keeps costs stable.

Commands
- Generate from a pack

    datagap-report gen --pack packs/pack1 --out report_out \
      --llm on --model gpt-5-mini --embed text-embedding-3-small \
      --api-base https://api.openai.com

- Validate outputs

    datagap-report validate --out report_out

- Print a sample of backlog items

    datagap-report print-sample --out report_out

Environment
- Requires Python 3.11+.
- Set `LLM_API_KEY` for API auth. Override base with `--api-base`.

Models
- Default LLM: `gpt-5-mini` (override with `--model gpt-5`).
- Embeddings: `text-embedding-3-small` (override with `--embed text-embedding-3-large`).
- Structured Outputs: enforced via JSON Schema with `strict: true` using the Responses API.

Docs
- OpenAI Models: https://platform.openai.com/docs/models
- Structured Outputs (Responses API): https://platform.openai.com/docs/guides/structured-outputs

Determinism and Seeds
- By default generation is non-deterministic.
- Pass `--seed <int>` to make outputs byte-identical across runs. When a seed is provided, the generator derives a deterministic “now” timestamp from the seed so all timestamps and IDs are stable.

Outputs
Packs are written as:

    packs/<pack_id>/
      manifest.json
      hashes.json
      catalog/
        datasets.json
        freshness.csv
        lineage.json
      nl_queries.jsonl
      slack.jsonl
      email.jsonl

Schemas
- `datagap-synth init` writes `schemas/` with JSON Schemas and a starter `config.yaml`.
- `validate` uses `schemas/` in CWD if present, or falls back to embedded copies.
