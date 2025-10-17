"""Domain generator abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Protocol

import numpy as np
import pandas as pd

from ..config import Config


class DomainGenerator(Protocol):
    name: str

    def generate_clean(self, cfg: Config, rng: np.random.Generator) -> dict[str, pd.DataFrame]: ...

    def context_fn(self, table: str) -> Callable[[pd.Series], Dict[str, str]]: ...

    def source_for(self, table: str) -> str: ...

    def business_key(self, table: str) -> list[str]: ...


@dataclass(slots=True)
class BaseDomain:
    name: str
    table_sources: Dict[str, str]

    def context_fn(self, table: str) -> Callable[[pd.Series], Dict[str, str]]:
        def _ctx(row: pd.Series) -> Dict[str, str]:
            ctx: Dict[str, str] = {}
            for field in ("platform", "channel", "device_type", "customer_type", "plan", "segment"):
                if field in row.index and pd.notna(row[field]):
                    ctx[field] = str(row[field])
            return ctx

        return _ctx

    def source_for(self, table: str) -> str:
        return self.table_sources.get(table, next(iter(self.table_sources.values())))

    def business_key(self, table: str) -> list[str]:
        if table.startswith("dim_"):
            return ["id"]
        return ["id"]


def get_domain(domain: str) -> DomainGenerator:
    if domain == "fintech":
        from .fintech import FintechDomain

        return FintechDomain()
    if domain == "ecom":
        from .ecom import EcomDomain

        return EcomDomain()
    if domain == "saas":
        from .saas import SaasDomain

        return SaasDomain()
    raise ValueError(f"Unsupported domain: {domain}")
