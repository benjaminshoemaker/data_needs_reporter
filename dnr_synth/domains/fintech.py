"""Fintech domain data generation."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import deterministic_faker
from .base import BaseDomain


class FintechDomain(BaseDomain):
    def __init__(self) -> None:
        super().__init__(
            name="fintech",
            table_sources={
                "dim_customer": "app_db",
                "dim_merchant": "app_db",
                "fct_transaction": "events",
                "fct_auth_event": "events",
                "evt_checkout_step": "events",
            },
        )

    def business_key(self, table: str) -> list[str]:
        mapping = {
            "dim_customer": ["customer_id", "valid_from"],
            "dim_merchant": ["merchant_id"],
            "fct_transaction": ["transaction_id"],
            "fct_auth_event": ["auth_id"],
            "evt_checkout_step": ["event_id"],
        }
        return mapping.get(table, super().business_key(table))

    def generate_clean(self, cfg: Config, rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
        faker = deterministic_faker(int(rng.integers(0, 10_000)))
        size_scale = {"small": 0.05, "medium": 0.25, "large": 1.0}[cfg.size]

        customer_n = max(200, int(cfg.keys.get("customers", 50_000) * size_scale))
        merchant_n = max(50, int(cfg.keys.get("merchants", 1_200) * size_scale))

        start_dt = datetime.combine(cfg.clock.start, datetime.min.time())
        days = [start_dt + timedelta(days=offset) for offset in range(cfg.clock.days)]

        customers = self._build_customers(customer_n, days, faker, rng)
        merchants = self._build_merchants(merchant_n, faker, rng)
        transactions = self._build_transactions(days, customers, merchants, rng)
        auth_events = self._build_auth_events(transactions, rng)
        checkout = self._build_checkout_steps(transactions, rng)

        frames: Dict[str, pd.DataFrame] = {
            "dim_customer": customers,
            "dim_merchant": merchants,
            "fct_transaction": transactions,
            "fct_auth_event": auth_events,
            "evt_checkout_step": checkout,
        }
        return frames

    def _build_customers(
        self,
        count: int,
        days: List[datetime],
        faker,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        customer_ids = np.arange(1, count + 1)
        created_at = rng.choice(days, size=count)
        segments = rng.choice(["consumer", "smb", "enterprise"], size=count, p=[0.7, 0.25, 0.05])
        pref_platform = rng.choice(["web", "ios", "android"], size=count, p=[0.55, 0.3, 0.15])

        records = []
        for cust_id, created, segment, platform in zip(customer_ids, created_at, segments, pref_platform):
            records.append(
                {
                    "customer_sk": cust_id,
                    "customer_id": f"C{cust_id:06d}",
                    "full_name": faker.name(),
                    "email": faker.email(),
                    "segment": segment,
                    "platform_preference": platform,
                    "country": faker.country_code(representation="alpha-2"),
                    "valid_from": created,
                    "valid_to": pd.NaT,
                    "ds": created.date().isoformat(),
                }
            )

        df = pd.DataFrame.from_records(records)
        return df

    def _build_merchants(self, count: int, faker, rng: np.random.Generator) -> pd.DataFrame:
        merchant_ids = np.arange(1, count + 1)
        categories = [
            "retail",
            "food",
            "travel",
            "digital",
            "services",
        ]
        records = []
        for mid in merchant_ids:
            launched = faker.date_time_between(start_date="-5y", end_date="-30d")
            records.append(
                {
                    "merchant_sk": mid,
                    "merchant_id": f"M{mid:05d}",
                    "merchant_name": faker.company(),
                    "category": rng.choice(categories),
                    "country": faker.country_code(representation="alpha-2"),
                    "valid_from": launched,
                    "valid_to": pd.NaT,
                    "ds": launched.date().isoformat(),
                }
            )
        return pd.DataFrame.from_records(records)

    def _build_transactions(
        self,
        days: List[datetime],
        customers: pd.DataFrame,
        merchants: pd.DataFrame,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        rows: List[Dict] = []
        transaction_count = max(300, len(customers) * 4)
        customer_ids = customers["customer_id"].to_numpy()
        merchant_ids = merchants["merchant_id"].to_numpy()
        platforms = np.array(["web", "ios", "android"])

        for i in range(transaction_count):
            day = rng.choice(days)
            event_ts = day + timedelta(minutes=int(rng.integers(0, 24 * 60)))
            session = f"sess_{rng.integers(10**6, 10**7)}"
            device = f"dev_{rng.integers(10**6, 10**7)}"
            platform = rng.choice(platforms, p=[0.5, 0.3, 0.2])
            rows.append(
                {
                    "transaction_id": f"txn_{i:08d}",
                    "customer_id": rng.choice(customer_ids),
                    "merchant_id": rng.choice(merchant_ids),
                    "session_id": session,
                    "device_id": device,
                    "platform": platform,
                    "currency": rng.choice(["USD", "EUR", "GBP"]),
                    "status": rng.choice(["authorized", "declined", "refunded"]),
                    "total_amount": float(np.round(rng.gamma(shape=2.0, scale=80.0), 2)),
                    "device_type": rng.choice(["mobile", "desktop", "tablet"], p=[0.6, 0.3, 0.1]),
                    "event_time": event_ts,
                    "ingested_at": event_ts,
                    "ds": event_ts.date().isoformat(),
                }
            )

        return pd.DataFrame(rows)

    def _build_auth_events(self, transactions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        subset = transactions.sample(frac=0.6, random_state=rng.integers(0, 1_000_000))
        records = []
        for idx, row in subset.iterrows():
            ts = row["event_time"] - timedelta(minutes=int(rng.integers(1, 15)))
            records.append(
                {
                    "auth_id": f"auth_{idx:08d}",
                    "transaction_id": row["transaction_id"],
                    "customer_id": row["customer_id"],
                    "merchant_id": row["merchant_id"],
                    "channel": rng.choice(["in_app", "web", "api"]),
                    "result": rng.choice(["success", "failure"], p=[0.93, 0.07]),
                    "event_time": ts,
                    "ingested_at": ts,
                    "ds": ts.date().isoformat(),
                }
            )
        return pd.DataFrame.from_records(records)

    def _build_checkout_steps(self, transactions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        steps = ["cart", "shipping", "payment", "confirmation"]
        records = []
        for _, row in transactions.sample(frac=0.3, random_state=rng.integers(0, 1_000_000)).iterrows():
            base_time = row["event_time"] - timedelta(minutes=5)
            for i, step in enumerate(steps):
                ts = base_time + timedelta(minutes=i)
                records.append(
                    {
                        "event_id": f"chk_{row['transaction_id']}_{i}",
                        "transaction_id": row["transaction_id"],
                        "customer_id": row["customer_id"],
                        "session_id": row["session_id"],
                        "platform": row["platform"],
                        "step": step,
                        "event_time": ts,
                        "ingested_at": ts,
                        "ds": ts.date().isoformat(),
                    }
                )
        return pd.DataFrame.from_records(records)
