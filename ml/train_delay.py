"""Preparación y evaluación prospectiva del riesgo de retraso.

`python ml/train_delay.py --check-data` no ajusta modelos ni escribe artefactos.
Sin ese indicador se ejecutará el entrenamiento, solo con el bloque train.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

if __package__:
    from .common import MODELS, checked_join, delivered_orders, load, require_unique
    from .temporal import chronological_split
else:
    from common import MODELS, checked_join, delivered_orders, load, require_unique
    from temporal import chronological_split

NUM = ["precio", "flete", "peso_g", "n_items", "dias_estimados", "mes", "dia_semana", "mismo_estado"]
CAT = ["estado_cliente", "estado_vendedor"]


def build_dataset(
    orders=None, items=None, products=None, sellers=None, customers=None,
) -> pd.DataFrame:
    """Una fila por pedido; NaN de peso se conserva para HistGradientBoosting."""
    source_orders = load("orders") if orders is None else orders
    orders = delivered_orders(source_orders)
    audit = dict(orders.attrs["audit"])
    items = (load("order_items") if items is None else items).copy()
    products = load("products") if products is None else products
    sellers = load("sellers") if sellers is None else sellers
    customers = load("customers") if customers is None else customers
    require_unique(items, ["order_id", "order_item_id"], "Ítems")
    checked_join(items[["order_id"]], source_orders[["order_id"]], "order_id")
    enriched = checked_join(items, products[["product_id", "product_weight_g"]], "product_id")
    enriched = checked_join(enriched, sellers[["seller_id", "seller_state"]], "seller_id")
    for column in ("price", "freight_value", "product_weight_g"):
        enriched[column] = pd.to_numeric(enriched[column], errors="raise")
        if np.isinf(enriched[column]).any() or enriched[column].lt(0).any():
            raise ValueError(f"Valor negativo o infinito: {column}")
    if enriched[["price", "freight_value"]].isna().any().any():
        raise ValueError("No se pueden sumar importes faltantes")
    enriched["seller_state"] = enriched["seller_state"].astype("string").str.upper()
    if enriched["seller_state"].isna().any() or not enriched["seller_state"].str.fullmatch("[A-Z]{2}").all():
        raise ValueError("Estado de vendedor no representable por el contrato")
    enriched["product_weight_g"] = enriched["product_weight_g"].where(enriched["product_weight_g"].gt(0))
    agg = enriched.groupby("order_id").agg(
        precio=("price", "sum"), flete=("freight_value", "sum"),
        peso_g=("product_weight_g", "sum"), known_weights=("product_weight_g", "count"),
        n_items=("order_item_id", "size"), n_estados_vendedor=("seller_state", "nunique"),
    )
    # No se confunde peso parcial (o totalmente ausente) con peso total conocido.
    agg.loc[agg["known_weights"].lt(agg["n_items"]), "peso_g"] = np.nan
    # Estado dominante por importe en centavos, luego artículos y sigla ascendente.
    enriched["_cents"] = enriched["price"].mul(100).round().astype("int64")
    origins = enriched.groupby(["order_id", "seller_state"], as_index=False).agg(
        cents=("_cents", "sum"), item_count=("order_item_id", "size")
    ).sort_values(
        ["order_id", "cents", "item_count", "seller_state"],
        ascending=[True, False, False, True], kind="stable",
    ).drop_duplicates("order_id").set_index("order_id")
    agg["estado_vendedor"] = origins["seller_state"]
    frame = checked_join(
        orders, customers[["customer_id", "customer_unique_id", "customer_state"]], "customer_id"
    )
    frame = checked_join(frame, agg.reset_index(), "order_id")
    frame["estado_cliente"] = frame["customer_state"].astype("string").str.upper()
    if frame["estado_cliente"].isna().any() or not frame["estado_cliente"].str.fullmatch("[A-Z]{2}").all():
        raise ValueError("Estado de cliente no representable por el contrato")
    frame["dias_estimados"] = (
        frame["order_estimated_delivery_date"].dt.normalize()
        - frame["order_purchase_timestamp"].dt.normalize()
    ).dt.days
    frame["mes"] = frame["order_purchase_timestamp"].dt.month
    frame["dia_semana"] = frame["order_purchase_timestamp"].dt.day_of_week
    frame["mismo_estado"] = frame["estado_cliente"].eq(frame["estado_vendedor"]).astype(int)
    audit.update({
        "joined_orders": len(frame),
        "unknown_or_zero_weight_orders": int(frame["peso_g"].isna().sum()),
        "multi_state_orders": int(frame["n_estados_vendedor"].gt(1).sum()),
        "late_calendar_day": int(frame["retrasado"].sum()),
    })
    # Metadatos para auditar y partir, nunca suministrados al estimador.
    frame.attrs["audit"] = audit
    return frame


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
             ("num", "passthrough", NUM)],
            remainder="drop", sparse_threshold=0,
        )),
        # Manejo nativo de NaN; sin imputación global ni validación interna aleatoria.
        ("clf", HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=300, random_state=42, early_stopping=False,
        )),
    ])


def main(check_only=False, **cutoffs):
    frame = build_dataset()
    split = chronological_split(frame, **cutoffs)
    for name in ("train", "validation", "test"):
        if getattr(split, name)["retrasado"].nunique() != 2:
            raise ValueError(f"{name} no contiene ambas clases; no cambiar a un split aleatorio")
    summary = {"preparation": frame.attrs["audit"], "temporal_split": split.summary()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if check_only:
        return summary
    model = build_pipeline()
    model.fit(split.train[NUM + CAT], split.train["retrasado"])
    for name in ("validation", "test"):
        part = getattr(split, name)
        y = part["retrasado"]
        proba = model.predict_proba(part[NUM + CAT])[:, 1]
        print(f"{name}: ROC-AUC={roc_auc_score(y, proba):.4f} "
              f"PR-AUC(AP)={average_precision_score(y, proba):.4f} "
              f"Brier={brier_score_loss(y, proba):.4f}")
        print(classification_report(y, proba >= .5, digits=3, zero_division=0))
    # No se reajusta usando validation/test después de informar sus métricas.
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / "delay_model.joblib")
    print("Guardado en", MODELS / "delay_model.joblib")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--train-end", default="2018-05-01")
    parser.add_argument("--validation-start", default="2018-06-01")
    parser.add_argument("--validation-end", default="2018-07-01")
    parser.add_argument("--test-start", default="2018-08-01")
    parser.add_argument("--test-end", default="2018-09-01")
    parser.add_argument("--observation-end", default="2018-10-18")
    args = parser.parse_args()
    main(
        check_only=args.check_data, train_end=args.train_end,
        validation_start=args.validation_start, validation_end=args.validation_end,
        test_start=args.test_start, test_end=args.test_end, observation_end=args.observation_end,
    )
