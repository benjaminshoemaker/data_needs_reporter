"""Configuration models and helpers for dnr-synth."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, validator

from .utils import load_yaml


class Clock(BaseModel):
    tz: str
    start: date
    days: int


class IngestLag(BaseModel):
    mean: float
    sd: float


class SourceSpec(BaseModel):
    ingest_lag_min: IngestLag
    duplicate_rate: float | None = 0.0

    model_config = ConfigDict(extra="allow")


class NullSpike(BaseModel):
    table: str
    field: str
    p: float = Field(ge=0.0, le=1.0)
    when: str
    where: str | None = None

    @validator("when")
    def validate_when(cls, value: str) -> str:
        parts = value.split("/")
        if len(parts) != 2:
            raise ValueError("when must be start/end date in ISO format")
        for part in parts:
            datetime.strptime(part, "%Y-%m-%d")
        return value


class Drift(BaseModel):
    at: date
    table: str
    rename: Dict[str, str] | None = None
    add: Dict[str, str] | None = None


class Outputs(BaseModel):
    warehouse: str
    dbt_project_dir: str


class Config(BaseModel):
    """Top-level configuration model parsed from YAML."""

    domain: Literal["fintech", "ecom", "saas"]
    size: Literal["small", "medium", "large"]
    clock: Clock
    keys: Dict[str, int]
    sources: Dict[str, SourceSpec]
    null_spikes: list[NullSpike] = Field(default_factory=list)
    schema_drift: list[Drift] = Field(default_factory=list)
    key_presence: Dict[str, Dict[str, Dict[str, float]]] = Field(default_factory=dict)
    joinability_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    outputs: Outputs

    model_config = ConfigDict(extra="allow")


def load_config(path: Path) -> Config:
    """Load *path* and return a Config instance."""

    data = load_yaml(path)
    return Config.model_validate(data)


def infer_seed(config_path: Path, override: int | None) -> int:
    """Return a deterministic seed for a config path."""

    if override is not None:
        return override
    return abs(hash(config_path.resolve())) % (2**32)
