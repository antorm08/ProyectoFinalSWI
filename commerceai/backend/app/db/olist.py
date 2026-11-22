"""Acceso a los CSV de Olist (capa de datos).

Más adelante puede sustituirse por PostgreSQL/SQLite sin cambiar los servicios.
"""
from functools import lru_cache

import pandas as pd

from app.core.config import OLIST_DIR

FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "customers": "olist_customers_dataset.csv",
}


def available(name: str) -> bool:
    return (OLIST_DIR / FILES[name]).exists()


@lru_cache(maxsize=None)
def load(name: str) -> pd.DataFrame:
    return pd.read_csv(OLIST_DIR / FILES[name])
