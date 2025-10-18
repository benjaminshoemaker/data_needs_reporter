"""Typer CLI for dnr-synth."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import logging
import pandas as pd
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from . import __version__
from .config import Config, load_config
from .corruptors import duplicates, keys, lateness, null_spikes, schema_drift
from .domains.base import DomainGenerator, get_domain
from .eval.metrics import health_metrics
from .eval.report import write_reports
from .writers.parquet import write_partitioned
from .utils import coerce_local_path, ensure_dir, get_rng, load_yaml, progress
from .sample import (
    build_context,
    generate_queries,
    generate_threads,
    validate_queries,
    validate_threads,
    write_json,
)
from .analysis.gaps_prompt import build_prompt, run_llm, split_outputs

console = Console()

app = typer.Typer(add_completion=False, help="Generate messy synthetic datasets for analytics workflows.")
log = logging.getLogger("dnr_synth.cli")

ARTIFACT_FILES_TO_KEEP = {
    "PROMPT_NL_QUERIES.md",
    "PROMPT_SLACK_THREADS.md",
    "nl_queries.json",
    "slack_threads.json",
}


@app.callback()
def _meta(version: bool = typer.Option(False, "--version", help="Show version and exit")) -> None:
    if version:
        console.print(f"dnr-synth {__version__}")
        raise typer.Exit()


@app.command()
def init(
    domain: str = typer.Argument(..., help="Domain to scaffold (fintech|ecom|saas)"),
) -> None:
    """Scaffold configs, dbt project, and artifacts directory."""

    domain = domain.lower()
    if domain not in {"fintech", "ecom", "saas"}:
        raise typer.BadParameter("domain must be one of fintech|ecom|saas")

    root = Path.cwd()
    domain_dir = root / "domains" / domain
    config_path = domain_dir / "config.yaml"
    dbt_dir = root / f"dbt_{domain}"
    artifacts_dir = root / "artifacts" / domain

    _write_domain_config(domain, config_path)
    _copy_dbt_template(domain, dbt_dir)
    ensure_dir(artifacts_dir)

    console.print(f"Initialized domain '{domain}'")
    console.print(f"- config: {config_path}")
    console.print(f"- dbt project: {dbt_dir}")
    console.print(f"- artifacts: {artifacts_dir}")


@app.command()
def generate(
    config: Path = typer.Option(..., "--config", help="Path to config.yaml"),
    out: Optional[Path] = typer.Option(None, "--out", help="Output directory for parquet data"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Deterministic seed"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan without writing data"),
) -> None:
    """Generate synthetic datasets for the config."""

    cfg = load_config(config)
    generator = get_domain(cfg.domain)
    rng = get_rng(seed)
    out_dir = coerce_local_path(out or cfg.outputs.warehouse)

    if dry_run:
        table = Table(title="Generation Plan", header_style="bold", box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Item")
        table.add_column("Value")
        table.add_row("Domain", cfg.domain)
        table.add_row("Tables", ", ".join(generator.table_sources.keys()))
        table.add_row("Output", str(out_dir))
        table.add_row("Seed", str(seed or 7))
        console.print(table)
        raise typer.Exit()

    with progress("Generating base datasets"):
        frames = generator.generate_clean(cfg, rng)

    frames = _apply_corruptors(frames, cfg, generator, rng)
    frames = schema_drift.apply(frames, cfg.schema_drift)

    with progress("Writing parquet outputs"):
        write_partitioned(frames, out_dir)

    console.print(f"Wrote datasets to {out_dir}")


@app.command()
def evaluate(
    domain: str = typer.Option(..., "--domain", help="Domain to evaluate"),
    data: Optional[Path] = typer.Option(None, "--data", help="Directory containing parquet tables"),
    out: Optional[Path] = typer.Option(None, "--out", help="Artifacts directory"),
) -> None:
    """Evaluate data health and write reports."""

    domain = domain.lower()
    base_data = Path(data) if data else Path("data") / domain
    if not base_data.exists():
        raise typer.BadParameter(f"Data directory {base_data} does not exist")

    frames = _load_parquet_frames(base_data)
    metrics = health_metrics(frames)
    out_dir = out or Path("artifacts") / domain
    write_reports(metrics, out_dir)
    console.print(f"Wrote evaluation artifacts to {out_dir}")


@app.command()
def sample(
    domain: str = typer.Option(..., "--domain", help="Domain for sampling"),
    data: Path = typer.Option(..., "--data", help="Path to parquet data directory"),
    dbt: Path = typer.Option(..., "--dbt", help="Path to dbt project directory"),
    out: Optional[Path] = typer.Option(None, "--out", help="Output directory for artifacts"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Deterministic seed"),
) -> None:
    """Generate grounded NL queries and Slack conversations."""

    domain = domain.lower()
    out_dir = out or Path("artifacts") / domain
    ensure_dir(out_dir)

    ctx = build_context(domain, str(data), str(dbt), seed or 7)

    queries = _generate_with_retries(lambda s: generate_queries(ctx, 60, s), validate_queries, ctx, 60, seed or 7)
    threads_raw = _generate_with_retries(lambda s: generate_threads(ctx, 6, s), validate_threads, ctx, 6, seed or 7)

    threads_payload = [
        {
            "thread_id": f"{domain}_thread_{i:02d}",
            "channel": f"#{domain}-data",
            "messages": thread,
        }
        for i, thread in enumerate(threads_raw)
    ]

    write_json(queries, out_dir / "nl_queries.json")
    write_json(threads_payload, out_dir / "slack_threads.json")
    console.print(f"Wrote grounded samples to {out_dir}")


def _generate_with_retries(generator, validator, ctx, target_count: int, seed: int) -> list:
    collected = []
    for attempt in range(3):
        current_seed = seed + attempt * 97
        raw = generator(current_seed)
        try:
            validated = validator(raw, ctx)
        except ValueError:
            continue
        collected.extend(validated)
        if len(collected) >= target_count:
            break
    if len(collected) < target_count:
        raise typer.BadParameter("Unable to generate sufficient validated samples")
    return collected[:target_count]


@app.command()
def export(
    to: str = typer.Option(..., "--to", help="Target sink (only postgres supported)"),
    dsn: str = typer.Option(..., "--dsn", help="Database connection string"),
) -> None:
    """Stub export command."""

    if to.lower() != "postgres":
        raise typer.BadParameter("Only --to postgres is currently supported")
    console.print("Export stub - no-op. TODO: implement Postgres writer.")
    console.print(f"Would connect using DSN: {dsn}")


@app.command()
def preview(
    folder: str = typer.Argument(..., help="Folder name under data/ to preview"),
    limit: int = typer.Option(25, help="Number of rows or items to preview"),
) -> None:
    """Show a rich preview of datasets, NL queries, and Slack threads."""

    base = Path("data") / folder
    if not base.exists():
        raise typer.BadParameter(f"Data folder {base} does not exist")

    _render_datasets(base, limit)
    _render_samples(folder, limit)


@app.command("clean")
def clean_command(
    domain: list[str] = typer.Option(None, "--domain", help="Domain(s) whose generated artifacts should be removed"),
    all_domains: bool = typer.Option(False, "--all", help="Remove data/artifacts for every domain"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
) -> None:
    """Remove generated data, artifacts, and reports so you can start fresh."""

    targets: list[Path] = []
    data_root = Path("data")
    artifacts_root = Path("artifacts")

    if all_domains:
        if data_root.exists():
            targets.extend(p for p in data_root.iterdir() if p.is_dir())
        if artifacts_root.exists():
            targets.extend(p for p in artifacts_root.iterdir() if p.is_dir())
    elif domain:
        for d in domain:
            if not d.strip():
                continue
            if data_root.exists():
                targets.append(data_root / d)
            if artifacts_root.exists():
                targets.append(artifacts_root / d)
    else:
        raise typer.BadParameter("Provide --domain or --all to specify what should be cleaned")

    extra_reports = [p for p in Path.cwd().glob("report*.md")]
    targets.extend(extra_reports)

    existing = [p for p in targets if p.exists()]
    if not existing:
        console.print("[yellow]Nothing to clean.")
        return

    console.print("The following paths will be removed:")
    for path in existing:
        console.print(f"  - {path}")

    if not yes:
        confirm = typer.confirm("Proceed with deletion?", abort=True)
        if not confirm:
            raise typer.Abort()

    for path in existing:
        try:
            if path.is_dir():
                if _is_artifact_domain_dir(path, artifacts_root):
                    _clean_artifact_domain_dir(path, ARTIFACT_FILES_TO_KEEP)
                else:
                    shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except Exception as exc:
            console.print(f"[red]Failed to remove {path}: {exc}")
        else:
            console.print(f"[green]Cleaned {path}")


def _is_artifact_domain_dir(path: Path, artifacts_root: Path) -> bool:
    try:
        return path.resolve().parent == artifacts_root.resolve()
    except FileNotFoundError:
        return path.parent == artifacts_root


def _clean_artifact_domain_dir(path: Path, keep_files: set[str]) -> None:
    if not path.exists():
        return
    for item in list(path.iterdir()):
        if item.is_file() and item.name in keep_files:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)
    remaining = list(path.iterdir()) if path.exists() else []
    if not remaining:
        path.rmdir()

def _extract_text_from_payload(raw: str) -> str | None:
    """Extract the 'text' or 'output_text' field from a Responses payload.
    Returns None if not found. Accepts raw JSON string or already-unwrapped text.
    """
    try:
        s = raw.strip()
        if not (s.startswith("{") or s.startswith("[")):
            return raw  # assume it's already plain text/markdown
        data = json.loads(s)
        # Fast-path known shapes
        if isinstance(data, dict):
            if isinstance(data.get("output_text"), str):
                return data["output_text"]
            if isinstance(data.get("content"), list):
                for ch in data["content"]:
                    txt = ch.get("text") if isinstance(ch, dict) else None
                    if isinstance(txt, str):
                        return txt
            if isinstance(data.get("response"), dict):
                inner = data["response"]
                if isinstance(inner.get("output_text"), str):
                    return inner["output_text"]
                if isinstance(inner.get("content"), list):
                    for ch in inner["content"]:
                        txt = ch.get("text") if isinstance(ch, dict) else None
                        if isinstance(txt, str):
                            return txt
        if isinstance(data, list):
            for msg in data:
                if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                    for ch in msg["content"]:
                        txt = ch.get("text") if isinstance(ch, dict) else None
                        if isinstance(txt, str):
                            return txt
        # Recursive search for any 'text' field
        def _find_text(obj):
            found: list[str] = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "text" and isinstance(v, str):
                        found.append(v)
                    else:
                        found.extend(_find_text(v))
            elif isinstance(obj, list):
                for it in obj:
                    found.extend(_find_text(it))
            return found
        texts = _find_text(data)
        if texts:
            # Prefer markdown-like (starts with '#' or contains '\n#')
            md_like = [t for t in texts if t.lstrip().startswith("#") or "\n#" in t]
            return (md_like[0] if md_like else max(texts, key=len))
        return None
    except Exception:
        return None


@app.command("gaps-text")
def gaps_text(
    path: Path = typer.Option(None, "--path", help="Path to raw LLM payload (defaults to artifacts/<domain>/gaps/llm_output.txt)"),
    domain: str = typer.Option(None, "--domain", help="Domain to infer default path when --path is not provided"),
) -> None:
    """Render the 'text' section from a raw LLM payload (Responses) as Markdown in the console."""
    if path is None:
        if not domain:
            raise typer.BadParameter("Provide --path or --domain")
        path = Path("artifacts") / domain / "gaps" / "llm_output.txt"
    if not path.exists():
        raise typer.BadParameter(f"Payload file not found at {path}")
    raw = path.read_text(encoding="utf-8")
    text = _extract_text_from_payload(raw)
    if not text:
        raise typer.BadParameter("Unable to find 'text' content in the payload")
    console.print(Markdown(text))


def _looks_like_gaps(items: object) -> bool:
    if not isinstance(items, list) or not items:
        return False
    good = 0
    for el in items:
        if not isinstance(el, dict):
            continue
        # Reject obvious wrapper messages
        if el.get("type") == "message" and "content" in el:
            return False
        # Accept if we see any gap-ish fields
        if any(k in el for k in ("gap_id", "gap_code", "title", "description")):
            good += 1
    return good > 0


@app.command("gaps-report")
def gaps_report(
    domain: str = typer.Option("fintech", "--domain", help="Domain name (e.g., fintech, ecom, saas)"),
    data: Path = typer.Option(None, "--data", help="Path to parquet data directory (default: data/<domain>)"),
    artifacts: Path = typer.Option(None, "--artifacts", help="Path to artifacts directory (default: artifacts/<domain>)"),
    out: Path = typer.Option(None, "--out", help="Output directory for report artifacts (default: artifacts/<domain>/gaps)"),
    emit_prompt: bool = typer.Option(True, "--emit-prompt", help="Write the constructed LLM prompt to PROMPT_GAPS.md"),
    run: bool = typer.Option(False, "--run", help="Call LLM (requires OPENAI_API_KEY or LLM_API_KEY)"),
    model: str = typer.Option("gpt-4o-mini", "--model", help="LLM model (Responses API)"),
    api_base: str = typer.Option("https://api.openai.com", "--api-base", help="LLM API base (Responses API)"),
    timeout_s: float = typer.Option(120.0, "--timeout-s", help="Timeout in seconds for the LLM request"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable info logging for this command"),
) -> None:
    """Build and optionally run an LLM-based gaps report over generated artifacts.

    By default, writes PROMPT_GAPS.md that you can paste into ChatGPT (offline-friendly).
    With --run, calls the Responses API and writes report.md + data_gaps_report.json if present.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    domain = domain.lower()
    data_dir = data or (Path("data") / domain)
    art_dir = artifacts or (Path("artifacts") / domain)
    out_dir = out or (art_dir / "gaps")
    ensure_dir(out_dir)

    if not data_dir.exists():
        raise typer.BadParameter(f"Data directory {data_dir} does not exist")
    if not art_dir.exists():
        raise typer.BadParameter(f"Artifacts directory {art_dir} does not exist")

    log.info(f"Building gaps prompt for domain={domain} data_dir={data_dir} artifacts_dir={art_dir}")
    prompt = build_prompt(domain, data_dir, art_dir)
    if emit_prompt:
        (out_dir / "PROMPT_GAPS.md").write_text(prompt, encoding="utf-8")
        log.info(f"Wrote prompt to {out_dir / 'PROMPT_GAPS.md'} (size={len(prompt)} chars)")

    if not run:
        console.print(f"Prompt written to {out_dir / 'PROMPT_GAPS.md'}")
        console.print("Run with --run to call the LLM, or paste into ChatGPT.")
        return

    # Resolve API key lazily to avoid env dependency if not running
    import os as _os
    # Best-effort: load .env.local if present
    try:
        env_path = Path(".env.local")
        if env_path.exists():
            log.info("Loading .env.local for API keys")
            for line in env_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip(); v = v.strip().strip('"').strip("'")
                if k and (k not in _os.environ or _os.environ[k] == ""):
                    _os.environ[k] = v
    except Exception:
        pass
    used_var = None
    api_key = None
    if _os.environ.get("LLM_API_KEY"):
        api_key = _os.environ.get("LLM_API_KEY")
        used_var = "LLM_API_KEY"
    elif _os.environ.get("OPENAI_API_KEY"):
        api_key = _os.environ.get("OPENAI_API_KEY")
        used_var = "OPENAI_API_KEY"
    if not api_key:
        raise typer.BadParameter("Missing API key. Set LLM_API_KEY or OPENAI_API_KEY.")
    log.info(f"Resolved API key from {used_var}")

    try:
        console.print("Calling LLM to generate report…")
        log.info(f"LLM request model={model} api_base={api_base} prompt_chars={len(prompt)}")
        raw = run_llm(prompt, model=model, api_base=api_base, api_key=api_key, timeout_s=timeout_s)
        log.info(f"LLM response received (size={len(raw)} chars)")
    except Exception as e:
        raise typer.BadParameter(f"LLM request failed: {e}")

    md, gaps = split_outputs(raw)
    text_payload = _extract_text_from_payload(raw)
    if md:
        (out_dir / "report.md").write_text(md, encoding="utf-8")
        console.print(f"Wrote markdown report: {out_dir / 'report.md'}")
        log.info(f"Wrote markdown report {out_dir / 'report.md'}")

    if text_payload:
        (out_dir / "report_text.md").write_text(text_payload, encoding="utf-8")
        log.info(f"Wrote raw text markdown {out_dir / 'report_text.md'}")

    # Only accept sidecar if it looks like a gaps list; otherwise try re-unwrapping and fallback
    final_gaps: list[dict] = []
    if gaps and _looks_like_gaps(gaps):
        final_gaps = gaps  # type: ignore
    else:
        md2, gaps2 = split_outputs(raw)
        if gaps2 and _looks_like_gaps(gaps2):
            final_gaps = gaps2  # type: ignore

    if final_gaps:
        write_json(final_gaps, out_dir / "data_gaps_report.json")
        console.print(f"Wrote JSON sidecar: {out_dir / 'data_gaps_report.json'}")
        log.info(f"Wrote JSON sidecar {out_dir / 'data_gaps_report.json'} with {len(final_gaps)} items")
        # Do not pretty-print or export; sidecar contains full details
    else:
        # Fallback: write raw output for inspection
        (out_dir / "llm_output.txt").write_text(raw, encoding="utf-8")
        console.print(f"[yellow]Wrote raw output: {out_dir / 'llm_output.txt'}")
        log.info(f"Wrote raw LLM output to {out_dir / 'llm_output.txt'}")


    # gaps-preview removed; focusing on report.md and JSON sidecar only


@app.command()
def prompts(
    domain: str = typer.Option(..., "--domain", help="Domain for prompts"),
    data: Path = typer.Option(..., "--data", help="Path to parquet data directory"),
    dbt: Path = typer.Option(..., "--dbt", help="Path to dbt project directory"),
    out: Optional[Path] = typer.Option(None, "--out", help="Artifacts directory for prompt files"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Deterministic seed for summaries"),
) -> None:
    """Emit ChatGPT-ready prompts to generate NL queries and Slack threads manually.

    Writes PROMPT_NL_QUERIES.md and PROMPT_SLACK_THREADS.md to the artifacts folder,
    and instructs where to save the ChatGPT outputs.
    """

    domain = domain.lower()
    out_dir = out or Path("artifacts") / domain
    ensure_dir(out_dir)
    ctx = build_context(domain, str(data), str(dbt), seed or 7)

    nl_path = out_dir / "PROMPT_NL_QUERIES.md"
    slack_path = out_dir / "PROMPT_SLACK_THREADS.md"
    nl_path.write_text(_compose_nl_prompt(ctx), encoding="utf-8")
    slack_path.write_text(_compose_slack_prompt(ctx), encoding="utf-8")

    console.print("Prompts written:")
    console.print(f"- NL queries: {nl_path}")
    console.print(f"- Slack threads: {slack_path}")
    console.print("After generating in ChatGPT, save outputs to:")
    console.print(f"- {out_dir / 'nl_queries.json'}")
    console.print(f"- {out_dir / 'slack_threads.json'}")


def _write_domain_config(domain: str, config_path: Path) -> None:
    ensure_dir(config_path.parent)
    config_examples = TEMPLATES_ROOT / "config_examples"
    if domain == "fintech":
        data = (config_examples / "fintech_minimal.yaml").read_text(encoding="utf-8")
        config_path.write_text(data, encoding="utf-8")
        return

    base = load_yaml(config_examples / "fintech_minimal.yaml")
    tweaked = _tweak_config_for_domain(domain, base)
    from ruamel.yaml import YAML

    yaml = YAML()
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.dump(tweaked, fh)


def _tweak_config_for_domain(domain: str, base: dict) -> dict:
    data = json.loads(json.dumps(base))
    data["domain"] = domain
    data["outputs"]["warehouse"] = f"parquet://./data/{domain}"
    data["outputs"]["dbt_project_dir"] = f"./dbt_{domain}"

    if domain == "ecom":
        data["null_spikes"] = [
            {"table": "fct_order", "field": "promotion_code", "p": 0.4, "when": "2025-08-05/2025-08-07", "where": "channel == 'web'"}
        ]
        data["schema_drift"] = [
            {"at": "2025-08-03", "table": "fct_order", "rename": {"unit_price": "unit_price_usd"}},
            {"at": "2025-08-12", "table": "dim_product", "add": {"color_family": "string"}},
        ]
        data["key_presence"] = {
            "fct_order": {
                "customer_id": {"base_p": 0.97, "store_p": 0.92},
                "product_id": {"base_p": 0.995},
            },
            "evt_clickstream": {
                "customer_id": {"base_p": 0.82},
                "session_id": {"base_p": 0.88},
            },
        }
        data["joinability_matrix"] = {
            "fct_order": {"dim_customer": 0.96, "dim_product": 0.94},
            "evt_clickstream": {"dim_customer": 0.8, "dim_product": 0.92},
        }
    elif domain == "saas":
        data["null_spikes"] = [
            {"table": "fct_usage_event_rollup", "field": "feature_adoption", "p": 0.3, "when": "2025-07-20/2025-07-22", "where": "account_id.str.endswith('5')"}
        ]
        data["schema_drift"] = [
            {"at": "2025-07-25", "table": "fct_subscription", "add": {"churn_reason": "string"}},
            {"at": "2025-07-28", "table": "dim_user", "rename": {"role": "user_role"}},
        ]
        data["key_presence"] = {
            "fct_subscription": {"account_id": {"base_p": 0.995}},
            "evt_product_usage": {
                "account_id": {"base_p": 0.9},
                "user_id": {"base_p": 0.87},
            },
        }
        data["joinability_matrix"] = {
            "fct_subscription": {"dim_account": 0.97},
            "evt_product_usage": {"dim_account": 0.9, "dim_user": 0.88},
        }
    return data


def _copy_dbt_template(domain: str, target_dir: Path) -> None:
    template_root = TEMPLATES_ROOT / "dbt_project"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(template_root, target_dir)
    for path in target_dir.rglob("*"):
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            content = content.replace("{{domain}}", domain)
            path.write_text(content, encoding="utf-8")


def _apply_corruptors(
    frames: dict[str, pd.DataFrame],
    cfg: Config,
    generator: DomainGenerator,
    rng,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for table, df in frames.items():
        ctx_fn = generator.context_fn(table)
        df1 = keys.apply_key_presence(df, cfg.key_presence.get(table, {}), ctx_fn, rng)
        fk_map = {f"dim_{col[:-3]}": col for col in df1.columns if col.endswith("_id")}
        df2 = keys.enforce_joinability(df1, cfg.joinability_matrix.get(table, {}), fk_map, rng)
        source = generator.source_for(table)
        source_cfg = cfg.sources.get(source)
        if source_cfg:
            df3 = lateness.add_lag(df2, source_cfg.ingest_lag_min.mean, source_cfg.ingest_lag_min.sd, rng)
            df4 = duplicates.add(df3, source_cfg.duplicate_rate or 0.0, (3, 60), rng)
        else:
            df4 = df2
        table_spikes = [s for s in cfg.null_spikes if s.table == table]
        df5 = null_spikes.apply(df4, table_spikes, rng) if table_spikes else df4
        result[table] = df5
    return result


def _load_parquet_frames(base_path: Path) -> dict[str, pd.DataFrame]:
    import pyarrow.dataset as ds

    frames: dict[str, pd.DataFrame] = {}
    for table_dir in base_path.iterdir():
        if not table_dir.is_dir():
            continue
        dataset = ds.dataset(str(table_dir), format="parquet", partitioning="hive")
        table = dataset.to_table()
        frames[table_dir.name] = table.to_pandas()
    if not frames:
        raise typer.BadParameter(f"No parquet tables found in {base_path}")
    return frames


def main() -> None:
    app()


if __name__ == "__main__":
    main()


def _render_datasets(base: Path, limit: int) -> None:
    import pyarrow.dataset as ds

    datasets = sorted([p for p in base.iterdir() if p.is_dir()])
    if not datasets:
        console.print(f"[yellow]No datasets found under {base}")
        return
    for table_dir in datasets:
        dataset = ds.dataset(str(table_dir), format="parquet", partitioning="hive")
        table = dataset.head(limit)
        df = table.to_pandas()
        rich_table = Table(title=f"{table_dir.name} (showing {len(df)} rows)")
        for col in df.columns:
            rich_table.add_column(str(col))
        for _, row in df.iterrows():
            rich_table.add_row(*[_stringify(row[col]) for col in df.columns])
        console.print(rich_table)


def _render_samples(folder: str, limit: int) -> None:
    artifacts = Path("artifacts") / folder
    nl_path = artifacts / "nl_queries.json"
    slack_path = artifacts / "slack_threads.json"

    if not artifacts.exists():
        console.print(f"[yellow]No artifacts folder found at {artifacts}")
        return

    if nl_path.exists():
        data = json.loads(nl_path.read_text())
        if data and "query" in data[0]:
            table = Table(title=f"NL Queries (showing {min(limit, len(data))} of {len(data)})")
            table.add_column("role")
            table.add_column("intent")
            table.add_column("time_range")
            table.add_column("query")
            table.add_column("refs")
            for entry in data[:limit]:
                table.add_row(
                    entry.get("role", ""),
                    entry.get("intent", ""),
                    entry.get("time_range", ""),
                    _truncate(entry.get("query", "")),
                    ", ".join(entry.get("references", [])[:3]),
                )
            console.print(table)
        else:
            table = Table(title=f"NL Queries (legacy format) showing {min(limit, len(data))}")
            table.add_column("id")
            table.add_column("prompt")
            for entry in data[:limit]:
                table.add_row(entry.get("id", ""), entry.get("prompt", ""))
            console.print(table)
    else:
        console.print(f"[yellow]No nl_queries.json at {nl_path}")

    if slack_path.exists():
        threads = json.loads(slack_path.read_text())[:limit]
        for thread in threads:
            messages = thread.get("messages", [])
            lines = []
            for msg in messages[:limit]:
                user = msg.get("user", "unknown")
                text = msg.get("text", "")
                ts = msg.get("ts", "")
                lines.append(f"[{user}] {text} ({ts})")
            body = "\n".join(lines)
            console.print(Panel(body, title=f"Slack {thread.get('thread_id', 'thread')}", subtitle=thread.get("channel", "")))
    else:
        console.print(f"[yellow]No slack_threads.json at {slack_path}")


def _stringify(value) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def _truncate(text: str, max_len: int = 90) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _compose_nl_prompt(ctx) -> str:
    tables = []
    for name, prof in ctx.tables.items():
        cols = ", ".join(prof.columns)
        tables.append(f"- {name}: {cols}")
    enums_preview = []
    for key, values in list(ctx.enums.items())[:15]:
        enums_preview.append(f"- {key}: {', '.join(values[:5])}")

    return f"""
You are ChatGPT. Generate a JSON array of 60 DATA-GROUNDED NL queries for the {ctx.domain} domain.

Requirements:
- Deterministic, clear English, grounded in the actual tables/columns listed below.
- Intent mix: descriptive 40%, comparative 25%, diagnostic 20%, anomaly 10%, forecast 5%.
- Roles include: Product Manager, Engineer, UX, Data Engineer, Data Analyst (blend across the set).
- Time ranges must fall within {ctx.calendar['start']} to {ctx.calendar['end']} (inclusive).
- Every item must reference existing columns with "table.col" in a references array; only use columns listed.

Output JSON only (no markdown), array of objects with fields:
  query: string
  role: one of ["Product Manager","Engineer","UX","Data Engineer","Data Analyst"]
  intent: one of ["descriptive","comparative","diagnostic","anomaly","forecast"]
  entities: object of domain-specific key/value pairs chosen from enums
  time_range: ISO date range like "YYYY-MM-DD to YYYY-MM-DD"
  references: array of strings with existing table.col entries

Tables and columns:
{chr(10).join(tables)}

Common categorical values (sample):
{chr(10).join(enums_preview)}

Only output JSON array. Do not include commentary.
"""


def _compose_slack_prompt(ctx) -> str:
    tables = []
    for name, prof in ctx.tables.items():
        cols = ", ".join(prof.columns[:8])
        tables.append(f"- {name}: {cols}")
    signals = []
    for a in ctx.signals.anomalies[:5]:
        signals.append(f"- anomaly: {a.metric} {a.direction} {a.delta:+.2f} on {a.when['start']}")
    for s in ctx.signals.null_spikes[:5]:
        signals.append(f"- null spike: {s.table}.{s.col} {s.null_rate:.2%} on {s.ds}")
    for l in ctx.signals.lag_spikes[:5]:
        signals.append(f"- lag spike: {l.table} p95={l.p95_minutes:.0f}m on {l.ds}")

    return f"""
You are ChatGPT. Generate a JSON array of 6 Slack threads grounded in the {ctx.domain} data.

Requirements:
- Each thread is 12–20 messages cycling roles at least once: PM, Data Analyst, Engineer, Data Engineer, UX.
- Use concrete references to existing columns (table.col) from the list below at least every third message.
- Include realistic stats (null rates, p95 lag, amounts) when applicable. Include 1–2 mitigations, one owner, and one follow-up task like ENG-123.
- Deterministic timestamps as increasing floats per thread, starting at a base, +60 seconds each message.
- Users (map roles to aliases): pm_riley, da_jules, eng_kai, de_amelia, ux_sol.

Output JSON only (no markdown). Array where each item is:
  thread_id: string
  channel: string (like "#{ctx.domain}-data")
  messages: array of objects with fields:
    ts: string (increasing float per message)
    user: one of ["pm_riley","da_jules","eng_kai","de_amelia","ux_sol"]
    text: markdown-compatible text that cites real table.col and concrete stats
    thread_ts: string (same as first message ts)

Tables and columns:
{chr(10).join(tables)}

Recent signals (if any):
{chr(10).join(signals) if signals else '- none detected; focus on data quality, nulls, and lag using the available fields.'}

Only output JSON array. Do not include commentary.
"""
TEMPLATES_ROOT = Path(__file__).resolve().parent.parent / "templates"
