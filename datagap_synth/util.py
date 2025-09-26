from __future__ import annotations

import csv
import hashlib
import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from importlib import resources


def make_rng(seed: int | None) -> random.Random:
    return random.Random(seed) if seed is not None else random.Random()


def deterministic_now(seed: int | None) -> datetime:
    if seed is None:
        return datetime.now(UTC)
    base = datetime(2023, 1, 1, tzinfo=UTC)
    # Spread seeds over ~5 years window deterministically
    seconds = (seed % (5 * 365 * 24 * 3600))
    return base + timedelta(seconds=seconds)


def isoformat_z(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def bounded_time(rng: random.Random, now: datetime, days: int) -> datetime:
    delta_seconds = rng.uniform(0, days * 24 * 3600)
    return now - timedelta(seconds=delta_seconds)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True))
            f.write("\n")
            count += 1
    return count


def write_csv(path: Path, headers: List[str], rows: Iterable[Tuple]) -> int:
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)
            count += 1
    return count


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_file_hashes(root: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "hashes.json":
            rel = p.relative_to(root).as_posix()
            mapping[rel] = sha256_file(p)
    return mapping


def write_schemas_to_dir(target_dir: Path) -> List[Path]:
    wrote: List[Path] = []
    for name in ["nl_query.schema.json", "slack.schema.json", "email.schema.json"]:
        with resources.files("datagap_synth.schemas").joinpath(name).open("rb") as src:
            data = src.read()
        out = target_dir / name
        out.write_bytes(data)
        wrote.append(out)
    return wrote


def sorted_by_when_id(records: List[dict]) -> List[dict]:
    return sorted(records, key=lambda r: (r.get("when", ""), r.get("id", "")))

