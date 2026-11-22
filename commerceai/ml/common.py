"""Utilidades compartidas por los scripts de entrenamiento."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLIST = ROOT / "data" / "olist"
MODELS = ROOT / "backend" / "app" / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def load(name: str) -> pd.DataFrame:
    path = OLIST / f"olist_{name}_dataset.csv"
    if not path.exists():
        raise SystemExit(f"Falta {path}. Descarga el dataset de Olist (ver data/README.md).")
    return pd.read_csv(path)


def delivered_orders() -> pd.DataFrame:
    """Pedidos entregados con fechas parseadas y una fila por pedido."""
    orders = load("orders")
    for c in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders[c] = pd.to_datetime(orders[c])
    orders = orders[orders["order_status"] == "delivered"].dropna(
        subset=["order_delivered_customer_date", "order_estimated_delivery_date"])
    orders["retrasado"] = (orders["order_delivered_customer_date"] >
                           orders["order_estimated_delivery_date"]).astype(int)
    orders["dias_entrega"] = (orders["order_delivered_customer_date"] -
                              orders["order_purchase_timestamp"]).dt.days
    return orders
