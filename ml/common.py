"""Carga y reglas de preparación compartidas; importar no escribe artefactos."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLIST = ROOT / "data" / "olist"
MODELS = ROOT / "backend" / "app" / "models"


def load(name: str) -> pd.DataFrame:
    path = OLIST / f"olist_{name}_dataset.csv"
    if not path.exists():
        raise SystemExit(f"Falta {path}. Descarga el dataset de Olist (ver data/README.md).")
    return pd.read_csv(path)


def require_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    """Fallar ante claves inválidas, en lugar de multiplicar o perder filas."""
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{name}: claves nulas o repetidas en {keys}")


def checked_join(left: pd.DataFrame, right: pd.DataFrame, on: str) -> pd.DataFrame:
    """Join N:1 completo: no descarta huérfanos ni multiplica filas."""
    require_unique(right, [on], f"Tabla de referencia {on}")
    if left[on].isna().any():
        raise ValueError(f"Claves nulas en {on}")
    result = left.merge(right, on=on, how="left", validate="many_to_one", indicator="_relation")
    if len(result) != len(left) or result["_relation"].ne("both").any():
        raise ValueError(f"Relación incompleta o multiplicada en {on}")
    return result.drop(columns="_relation")


def delivered_orders(source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Objetivo por día calendario; tiempos transcurridos conservan las horas."""
    orders = (load("orders") if source is None else source).copy()
    require_unique(orders, ["order_id"], "Pedidos")
    for c in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders[c] = pd.to_datetime(orders[c], errors="raise")
    eligible = orders["order_status"].eq("delivered")
    missing_target = eligible & orders[["order_delivered_customer_date", "order_estimated_delivery_date"]].isna().any(axis=1)
    orders = orders.loc[eligible & ~missing_target].copy()
    if orders["order_purchase_timestamp"].isna().any():
        raise ValueError("Un pedido elegible no tiene fecha de compra")
    actual = orders["order_delivered_customer_date"]
    purchase = orders["order_purchase_timestamp"]
    promised = orders["order_estimated_delivery_date"]
    if actual.lt(purchase).any() or promised.dt.normalize().lt(purchase.dt.normalize()).any():
        raise ValueError("Entrega o día prometido anterior a la compra")
    orders["retrasado"] = actual.dt.normalize().gt(promised.dt.normalize()).astype(int)
    orders["dias_entrega"] = (orders["order_delivered_customer_date"] -
                               orders["order_purchase_timestamp"]).dt.total_seconds() / 86400
    # Proxy conservador de disponibilidad: la etiqueta no se usa antes de la entrega.
    orders["label_available_at"] = actual
    orders.attrs["audit"] = {
        "source_orders": int(len(source) if source is not None else len(eligible)),
        "delivered_status": int(eligible.sum()),
        "missing_target_dates": int(missing_target.sum()),
        "eligible_orders": len(orders),
        "late_calendar_day": int(orders["retrasado"].sum()),
    }
    return orders
