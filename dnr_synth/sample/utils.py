"""Utility helpers for sampling outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
