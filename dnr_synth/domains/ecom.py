"""Ecommerce domain generator."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import deterministic_faker
from .base import BaseDomain


class EcomDomain(BaseDomain):
    def __init__(self) -> None:
        super().__init__(
            name="ecom",
            table_sources={
                "dim_customer": "app_db",
                "dim_product": "catalog",
                "fct_order": "events",
                "fct_payment": "events",
                "evt_clickstream": "events",
            },
        )

    def business_key(self, table: str) -> list[str]:
        mapping = {
            "dim_customer": ["customer_id", "valid_from"],
            "dim_product": ["product_id"],
            "fct_order": ["order_id"],
            "fct_payment": ["payment_id"],
            "evt_clickstream": ["event_id"],
        }
        return mapping.get(table, super().business_key(table))

    def generate_clean(self, cfg: Config, rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
        faker = deterministic_faker(int(rng.integers(0, 10_000)))
        size_scale = {"small": 0.05, "medium": 0.2, "large": 1.0}[cfg.size]

        customer_n = max(200, int(cfg.keys.get("customers", 40_000) * size_scale))
        product_n = max(150, int(cfg.keys.get("products", 5_000) * size_scale))

        start_dt = datetime.combine(cfg.clock.start, datetime.min.time())
        days = [start_dt + timedelta(days=offset) for offset in range(cfg.clock.days)]

        customers = self._build_customers(customer_n, days, faker, rng)
        products = self._build_products(product_n, faker, rng)
        orders = self._build_orders(days, customers, products, rng)
        payments = self._build_payments(orders, rng)
        clicks = self._build_clickstream(days, customers, products, rng)

        return {
            "dim_customer": customers,
            "dim_product": products,
            "fct_order": orders,
            "fct_payment": payments,
            "evt_clickstream": clicks,
        }

    def _build_customers(self, count: int, days: List[datetime], faker, rng: np.random.Generator) -> pd.DataFrame:
        records = []
        for cid in range(1, count + 1):
            created = rng.choice(days)
            records.append(
                {
                    "customer_sk": cid,
                    "customer_id": f"EC{cid:06d}",
                    "full_name": faker.name(),
                    "email": faker.email(),
                    "lifecycle_stage": rng.choice(["new", "active", "churned"], p=[0.2, 0.7, 0.1]),
                    "preferred_channel": rng.choice(["web", "mobile", "store"], p=[0.65, 0.3, 0.05]),
                    "valid_from": created,
                    "valid_to": pd.NaT,
                    "ds": created.date().isoformat(),
                }
            )
        return pd.DataFrame.from_records(records)

    def _build_products(self, count: int, faker, rng: np.random.Generator) -> pd.DataFrame:
        categories = ["apparel", "electronics", "home", "beauty", "outdoors"]
        records = []
        for pid in range(1, count + 1):
            launch = faker.date_time_between(start_date="-3y", end_date="-7d")
            records.append(
                {
                    "product_sk": pid,
                    "product_id": f"P{pid:06d}",
                    "name": faker.catch_phrase(),
                    "category": rng.choice(categories),
                    "base_price": float(np.round(rng.uniform(5.0, 350.0), 2)),
                    "valid_from": launch,
                    "valid_to": pd.NaT,
                    "ds": launch.date().isoformat(),
                }
            )
        return pd.DataFrame.from_records(records)

    def _build_orders(
        self,
        days: List[datetime],
        customers: pd.DataFrame,
        products: pd.DataFrame,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        rows: List[Dict] = []
        qty_choices = np.arange(1, 6)
        for i in range(max(400, len(customers) * 3)):
            day = rng.choice(days)
            event_ts = day + timedelta(minutes=int(rng.integers(0, 24 * 60)))
            customer = rng.choice(customers["customer_id"].to_numpy())
            product = rng.choice(products["product_id"].to_numpy())
            quantity = int(rng.choice(qty_choices, p=[0.45, 0.3, 0.15, 0.07, 0.03]))
            rows.append(
                {
                    "order_id": f"ord_{i:08d}",
                    "customer_id": customer,
                    "channel": rng.choice(["web", "mobile", "store"], p=[0.6, 0.3, 0.1]),
                    "country": rng.choice(["US", "CA", "GB", "DE", "AU"]),
                    "product_id": product,
                    "quantity": quantity,
                    "unit_price": float(np.round(rng.normal(45.0, 12.0), 2)),
                    "promotion_code": rng.choice([None, "SUMMER", "WELCOME", "VIP"], p=[0.6, 0.2, 0.15, 0.05]),
                    "event_time": event_ts,
                    "ingested_at": event_ts,
                    "ds": event_ts.date().isoformat(),
                }
            )
        return pd.DataFrame(rows)

    def _build_payments(self, orders: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        records = []
        payment_methods = ["credit_card", "paypal", "gift_card", "klarna"]
        for idx, row in orders.iterrows():
            pay_ts = row["event_time"] + timedelta(minutes=int(rng.integers(1, 20)))
            records.append(
                {
                    "payment_id": f"pay_{idx:08d}",
                    "order_id": row["order_id"],
                    "customer_id": row["customer_id"],
                    "amount": float(np.round(row["quantity"] * row["unit_price"], 2)),
                    "method": rng.choice(payment_methods, p=[0.65, 0.2, 0.1, 0.05]),
                    "status": rng.choice(["captured", "failed", "pending"], p=[0.92, 0.05, 0.03]),
                    "event_time": pay_ts,
                    "ingested_at": pay_ts,
                    "ds": pay_ts.date().isoformat(),
                }
            )
        return pd.DataFrame.from_records(records)

    def _build_clickstream(
        self,
        days: List[datetime],
        customers: pd.DataFrame,
        products: pd.DataFrame,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        records = []
        events = ["view", "add_to_cart", "wishlist", "checkout"]
        for _ in range(max(600, len(customers) * 4)):
            day = rng.choice(days)
            ts = day + timedelta(minutes=int(rng.integers(0, 24 * 60)))
            product = rng.choice(products["product_id"].to_numpy())
            customer = rng.choice(customers["customer_id"].to_numpy())
            records.append(
                {
                    "event_id": f"clk_{rng.integers(10**8)}",
                    "customer_id": customer,
                    "product_id": product,
                    "session_id": f"sess_{rng.integers(10**6, 10**7)}",
                    "event_type": rng.choice(events, p=[0.5, 0.2, 0.15, 0.15]),
                    "device_type": rng.choice(["desktop", "mobile", "tablet"], p=[0.4, 0.5, 0.1]),
                    "event_time": ts,
                    "ingested_at": ts,
                    "ds": ts.date().isoformat(),
                }
            )
        return pd.DataFrame.from_records(records)
