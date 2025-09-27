from __future__ import annotations

from importlib import resources
from typing import Any, Dict

import yaml


def load_scenario(name: str) -> Dict[str, Any]:
    pkg = "datagap_synth.scenarios"
    try:
        path = resources.files(pkg).joinpath(f"{name}.yaml")
        with path.open("rb") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Scenario YAML must be a mapping: {name}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Scenario not found: {name}")
