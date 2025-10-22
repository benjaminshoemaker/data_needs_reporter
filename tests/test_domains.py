from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from dnr_synth.config import Clock, Config, IngestLag, Outputs, SourceSpec
from dnr_synth.domains.ecom import EcomDomain
from dnr_synth.domains.fintech import FintechDomain
from dnr_synth.domains.saas import SaasDomain


def _sources() -> dict[str, SourceSpec]:
    names = {"app_db", "events", "catalog", "billing"}
    return {name: SourceSpec(ingest_lag_min=IngestLag(mean=5, sd=1), duplicate_rate=0.0) for name in names}


@pytest.mark.parametrize(
    "domain_cls, domain_name, keys, expected_tables",
    [
        (FintechDomain, "fintech", {"customers": 300, "merchants": 100}, {"dim_customer", "dim_merchant", "fct_transaction", "fct_auth_event", "evt_checkout_step"}),
        (EcomDomain, "ecom", {"customers": 200, "products": 80}, {"dim_customer", "dim_product", "fct_order", "fct_payment", "evt_clickstream"}),
        (SaasDomain, "saas", {"accounts": 160, "users": 600}, {"dim_account", "dim_user", "fct_subscription", "fct_usage_event_rollup", "evt_product_usage"}),
    ],
)
def test_domain_generate_clean_shapes(domain_cls, domain_name, keys, expected_tables):
    cfg = Config(
        domain=domain_name,
        size="small",
        clock=Clock(tz="UTC", start=date(2025, 1, 1), days=5),
        keys=keys,
        sources=_sources(),
        outputs=Outputs(warehouse="parquet://./tmp", dbt_project_dir="./dbt_tmp"),
    )

    domain = domain_cls()
    rng = np.random.default_rng(123)
    frames = domain.generate_clean(cfg, rng)

    assert set(frames) == expected_tables

    for table, df in frames.items():
        assert not df.empty, f"{table} should have rows"
        for key in domain.business_key(table):
            assert key in df.columns, f"{table} missing key column {key}"
            assert df[key].notna().any(), f"{table}.{key} is completely null"
