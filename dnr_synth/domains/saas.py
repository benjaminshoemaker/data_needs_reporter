"""SaaS domain generator."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import deterministic_faker
from .base import BaseDomain


class SaasDomain(BaseDomain):
    def __init__(self) -> None:
        super().__init__(
            name="saas",
            table_sources={
                "dim_account": "app_db",
                "dim_user": "app_db",
                "fct_subscription": "billing",
                "fct_usage_event_rollup": "events",
                "evt_product_usage": "events",
            },
        )

    def business_key(self, table: str) -> list[str]:
        mapping = {
            "dim_account": ["account_id", "valid_from"],
            "dim_user": ["user_id", "valid_from"],
            "fct_subscription": ["subscription_id"],
            "fct_usage_event_rollup": ["usage_id"],
            "evt_product_usage": ["event_id"],
        }
        return mapping.get(table, super().business_key(table))

    def generate_clean(self, cfg: Config, rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
        faker = deterministic_faker(int(rng.integers(0, 10_000)))
        size_scale = {"small": 0.05, "medium": 0.25, "large": 1.0}[cfg.size]

        account_n = max(120, int(cfg.keys.get("accounts", 12_000) * size_scale))
        user_n = max(400, int(cfg.keys.get("users", 80_000) * size_scale))

        start_dt = datetime.combine(cfg.clock.start, datetime.min.time())
        days = [start_dt + timedelta(days=offset) for offset in range(cfg.clock.days)]

        accounts = self._build_accounts(account_n, days, faker, rng)
        users = self._build_users(user_n, accounts, days, faker, rng)
        subscriptions = self._build_subscriptions(accounts, rng)
        usage_rollup = self._build_usage_rollups(days, accounts, rng)
        usage_events = self._build_usage_events(days, users, rng)

        return {
            "dim_account": accounts,
            "dim_user": users,
            "fct_subscription": subscriptions,
            "fct_usage_event_rollup": usage_rollup,
            "evt_product_usage": usage_events,
        }

    def _build_accounts(
        self,
        count: int,
        days: List[datetime],
        faker,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        plans = ["starter", "growth", "enterprise"]
        records = []
        for aid in range(1, count + 1):
            created = rng.choice(days)
            plan = rng.choice(plans, p=[0.55, 0.35, 0.10])
            records.append(
                {
                    "account_sk": aid,
                    "account_id": f"AC{aid:06d}",
                    "account_name": faker.company(),
                    "plan": plan,
                    "region": rng.choice(["AMER", "EMEA", "APAC"], p=[0.5, 0.3, 0.2]),
                    "status": rng.choice(["active", "trial", "churned"], p=[0.82, 0.12, 0.06]),
                    "valid_from": created,
                    "valid_to": pd.NaT,
                    "ds": created.date().isoformat(),
                }
            )
        return pd.DataFrame.from_records(records)

    def _build_users(
        self,
        count: int,
        accounts: pd.DataFrame,
        days: List[datetime],
        faker,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        account_ids = accounts["account_id"].to_numpy()
        records = []
        for uid in range(1, count + 1):
            created = rng.choice(days)
            account_id = rng.choice(account_ids)
            records.append(
                {
                    "user_sk": uid,
                    "user_id": f"U{uid:07d}",
                    "account_id": account_id,
                    "full_name": faker.name(),
                    "role": rng.choice(["admin", "editor", "viewer"], p=[0.1, 0.45, 0.45]),
                    "timezone": rng.choice(["UTC", "US/Pacific", "US/Eastern", "Europe/Berlin", "Asia/Singapore"]),
                    "valid_from": created,
                    "valid_to": pd.NaT,
                    "ds": created.date().isoformat(),
                }
            )
        return pd.DataFrame.from_records(records)

    def _build_subscriptions(self, accounts: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        records = []
        for idx, row in accounts.iterrows():
            start_ts = row["valid_from"]
            mrr = float(np.round(rng.lognormal(mean=3.2, sigma=0.5), 2))
            records.append(
                {
                    "subscription_id": f"sub_{idx:07d}",
                    "account_id": row["account_id"],
                    "plan": row["plan"],
                    "status": row["status"],
                    "mrr": mrr,
                    "arr": float(np.round(mrr * 12, 2)),
                    "event_time": start_ts,
                    "ingested_at": start_ts,
                    "ds": start_ts.date().isoformat(),
                }
            )
        return pd.DataFrame.from_records(records)

    def _build_usage_rollups(
        self,
        days: List[datetime],
        accounts: pd.DataFrame,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        rows: List[Dict] = []
        account_ids = accounts["account_id"].to_numpy()
        for day in days:
            for account_id in rng.choice(account_ids, size=int(len(account_ids) * 0.7), replace=False):
                active = int(rng.integers(1, 500))
                rows.append(
                    {
                        "usage_id": f"usage_{account_id}_{day.date().isoformat()}",
                        "account_id": account_id,
                        "active_users": active,
                        "events_count": int(rng.integers(active, active * 120)),
                        "feature_adoption": float(np.round(rng.beta(2.5, 4.0), 4)),
                        "event_time": day + timedelta(hours=23, minutes=45),
                        "ingested_at": day + timedelta(hours=23, minutes=45),
                        "ds": day.date().isoformat(),
                    }
                )
        return pd.DataFrame(rows)

    def _build_usage_events(
        self,
        days: List[datetime],
        users: pd.DataFrame,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        feature_areas = ["dashboard", "automation", "analytics", "billing"]
        records = []
        user_ids = users["user_id"].to_numpy()
        account_lookup = users.set_index("user_id")["account_id"].to_dict()
        for _ in range(max(800, len(users) // 5)):
            day = rng.choice(days)
            ts = day + timedelta(minutes=int(rng.integers(0, 24 * 60)))
            user_id = rng.choice(user_ids)
            records.append(
                {
                    "event_id": f"usage_{rng.integers(10**10)}",
                    "account_id": account_lookup[user_id],
                    "user_id": user_id,
                    "feature_area": rng.choice(feature_areas),
                    "platform": rng.choice(["web", "desktop", "api"], p=[0.6, 0.3, 0.1]),
                    "event_time": ts,
                    "ingested_at": ts,
                    "ds": ts.date().isoformat(),
                }
            )
        return pd.DataFrame.from_records(records)
