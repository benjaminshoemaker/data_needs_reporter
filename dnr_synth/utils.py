"""Utility helpers used across dnr-synth modules."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable

import numpy as np
from faker import Faker
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from ruamel.yaml import YAML

console = Console()

_yaml_reader = YAML(typ="safe")
_yaml_writer = YAML()
_yaml_writer.default_flow_style = False


def ensure_dir(path: Path) -> None:
    """Create *path* if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""

    with path.open("r", encoding="utf-8") as fh:
        data = _yaml_reader.load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must be a mapping")
    return data


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Persist *data* as YAML to *path*."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        _yaml_writer.dump(data, fh)


def get_rng(seed: int | None) -> np.random.Generator:
    """Return a seeded numpy random generator (PCG64)."""

    if seed is None:
        seed = 7
    return np.random.Generator(np.random.PCG64(seed))


def deterministic_faker(seed: int | None) -> Faker:
    """Return a Faker instance seeded from *seed*."""

    if seed is None:
        seed = 7
    faker = Faker()
    Faker.seed(seed)
    faker.random.seed(seed)
    return faker


def daterange(start: datetime, days: int) -> Iterable[datetime]:
    """Yield *days* consecutive dates from *start*."""

    for offset in range(days):
        yield start + timedelta(days=offset)


def utc_now() -> datetime:
    """Return naive UTC *now* to keep outputs deterministic in tests."""

    return datetime.now(timezone.utc).replace(tzinfo=None)


@contextmanager
def progress(task_description: str) -> Generator[Progress, None, None]:
    """Context manager yielding a Rich progress instance."""

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress_bar:
        progress_bar.add_task(task_description, total=None)
        yield progress_bar


def apply_in_place(frames: Dict[str, Any], fn: Callable[[Any], Any]) -> Dict[str, Any]:
    """Apply *fn* to all values in *frames* and return the updated mapping."""

    return {name: fn(frame) for name, frame in frames.items()}


def coerce_local_path(uri: str | Path) -> Path:
    """Coerce a user provided path or parquet URI into a local Path."""

    if isinstance(uri, Path):
        return uri
    if uri.startswith("parquet://"):
        resolved = uri.replace("parquet://", "", 1)
        return Path(resolved)
    return Path(uri)
