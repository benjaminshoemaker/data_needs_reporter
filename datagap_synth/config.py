from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG_YAML = """
schema_version: 1.0.0
time_window_days: 7
seed: null
actors:
  - u_alex
  - u_riley
  - u_jamie
audience_weights: {Exec: 3, Ops: 2, Eng: 1}
nlq:
  count: 800
  success_rate: 0.65
  slow_rate: 0.10
  uncertain_rate: 0.05
  gaps_distribution:
    missing_asset: 0.10
    missing_column: 0.12
    grain_mismatch: 0.06
    type_mismatch: 0.05
    freshness_breach: 0.08
    joinability_issue: 0.05
    access_denied: 0.03
    docs_missing: 0.12
    performance_limit: 0.04
slack:
  threads: 60
  replies_mean: 4
email:
  threads: 20
  replies_mean: 3
catalog:
  domains: [orders, customers, products, marketing]
  mart_tables: 10
  staging_tables: 20
  freshness_sla_hours: [24, 48, 72]
  inject_freshness_breaches: 0.15
"""


DEFAULT_CONFIG = {
    "schema_version": "1.0.0",
    "time_window_days": 7,
    "seed": None,
    "scenario": "enterprise",
    "actors": ["u_alex", "u_riley", "u_jamie"],
    "audience_weights": {"Exec": 3, "Ops": 2, "Eng": 1},
    "nlq": {
        "count": 1500,
        "success_rate": 0.65,
        "slow_rate": 0.10,
        "uncertain_rate": 0.05,
        "gaps_distribution": {
            "missing_asset": 0.10,
            "missing_column": 0.12,
            "grain_mismatch": 0.06,
            "type_mismatch": 0.05,
            "freshness_breach": 0.08,
            "joinability_issue": 0.05,
            "access_denied": 0.03,
            "docs_missing": 0.12,
            "performance_limit": 0.04,
        },
        "languages": [{"lang": "en", "pct": 0.9}, {"lang": "es", "pct": 0.05}, {"lang": "fr", "pct": 0.05}],
        "noise": {
            "typos_pct": 0.07,
            "abbreviations_pct": 0.12,
            "vague_time_pct": 0.10,
            "oos_term_pct": 0.12,
        },
    },
    "slack": {"threads": 120, "replies_mean": 5},
    "email": {"threads": 40, "replies_mean": 3},
    "catalog": {
        "domains": ["orders", "customers", "products", "marketing", "finance"],
        "mart_tables": 30,
        "staging_tables": 50,
        "freshness_sla_hours": [24, 48, 72],
        "inject_freshness_breaches": 0.15,
    },
}


def write_default_config(path: Path) -> None:
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False)
    except Exception:
        # Fallback: static YAML so init works without PyYAML
        text = DEFAULT_CONFIG_YAML.lstrip()
    path.write_text(text)


def load_config(path: Path) -> Dict[str, Any]:
    import yaml  # local import to avoid hard dependency for `init`

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("config.yaml must be a mapping")
    return data
