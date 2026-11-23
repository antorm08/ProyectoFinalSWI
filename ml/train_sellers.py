"""4. Clustering de vendedores según su desempeño (K-Means).

Salidas: backend/app/models/seller_kmeans.joblib y seller_clusters.csv
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .common import MODELS, checked_join, delivered_orders, load, require_unique
else:
    from common import MODELS, checked_join, delivered_orders, load, require_unique

FEATURES = ["n_pedidos", "facturacion", "nota_media", "tasa_retraso", "dias_entrega_medio"]
MIN_PEDIDOS = 5


def seller_order_metrics(orders=None, items=None, reviews=None) -> pd.DataFrame:
    """Una fila vendedor-pedido, antes de resumir por vendedor."""
    source = load("orders") if orders is None else orders
    eligible = delivered_orders(source)
    audit = dict(eligible.attrs["audit"])
    items = (load("order_items") if items is None else items).copy()
    reviews = (load("order_reviews") if reviews is None else reviews).copy()
    require_unique(items, ["order_id", "order_item_id"], "Ítems")
    require_unique(reviews, ["review_id", "order_id"], "Reseñas")
    if items["seller_id"].isna().any():
        raise ValueError("Ítem sin vendedor")
    # Verificar referencias antes de filtrar explícitamente por elegibilidad.
    checked_join(items[["order_id"]], source[["order_id"]], "order_id")
    checked_join(reviews[["order_id"]], source[["order_id"]], "order_id")
    items["price"] = pd.to_numeric(items["price"], errors="raise")
    if items["price"].isna().any() or not np.isfinite(items["price"]).all() or items["price"].lt(0).any():
        raise ValueError("Precio faltante, negativo o infinito")
    reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="raise")
    if (~reviews["review_score"].between(1, 5) | reviews["review_score"].mod(1).ne(0)).any():
        raise ValueError("Nota de reseña fuera de 1–5 o no entera")
    selected = items.loc[items["order_id"].isin(eligible["order_id"])]
    # Artículos distintos suman ingresos; no multiplican nota ni entrega.
    pairs = selected.groupby(["seller_id", "order_id"], as_index=False).agg(
        facturacion=("price", "sum"), n_articulos=("order_item_id", "size")
    )
    pairs = checked_join(pairs, eligible[["order_id", "retrasado", "dias_entrega"]], "order_id")
    # Varias reseñas -> una media por pedido ANTES del join con vendedor-pedido.
    scores = reviews.groupby("order_id", as_index=False).agg(nota=("review_score", "mean"))
    before = len(pairs)
    pairs = pairs.merge(scores, on="order_id", how="left", validate="many_to_one")
    if len(pairs) != before or pairs.duplicated(["seller_id", "order_id"]).any():
        raise AssertionError("El join de reseñas multiplicó vendedor-pedido")
    audit.update({
        "source_item_rows": len(items), "eligible_item_rows": len(selected),
        "seller_order_rows": len(pairs), "rows_after_reviews_join": len(pairs),
        "seller_orders_without_review": int(pairs["nota"].isna().sum()),
    })
    pairs.attrs["audit"] = audit
    return pairs


def seller_metrics(orders=None, items=None, reviews=None) -> pd.DataFrame:
    pairs = seller_order_metrics(orders, items, reviews)
    metrics = pairs.groupby("seller_id").agg(
        n_pedidos=("order_id", "size"), facturacion=("facturacion", "sum"),
        nota_media=("nota", "mean"), tasa_retraso=("retrasado", "mean"),
        dias_entrega_medio=("dias_entrega", "mean"),
    )
    minimum = metrics["n_pedidos"].ge(MIN_PEDIDOS)
    # No inventar satisfacción si no existe ninguna reseña de ese vendedor.
    missing = metrics[FEATURES].isna().any(axis=1)
    result = metrics.loc[minimum & ~missing].copy()
    result.attrs["audit"] = {
        **pairs.attrs["audit"], "sellers_with_eligible_orders": len(metrics),
        "sellers_below_minimum": int((~minimum).sum()),
        "sellers_missing_metrics_after_minimum": int((minimum & missing).sum()),
        "eligible_sellers": len(result),
    }
    return result


def label_clusters(centers) -> dict[int, str]:
    """Pone nombres legibles a los clusters según sus centroides (escala original).

    Revisad las etiquetas tras ver los centroides impresos: es un punto de partida.
    """
    labels = {}
    for c, row in centers.iterrows():
        if row["tasa_retraso"] == centers["tasa_retraso"].max():
            labels[c] = "Logística problemática"
        elif row["facturacion"] == centers["facturacion"].max():
            labels[c] = "Vendedores estrella (alto volumen)"
        elif row["nota_media"] == centers["nota_media"].min():
            labels[c] = "Baja satisfacción"
        else:
            labels[c] = "Vendedores estables"
    return {int(k): v for k, v in labels.items()}


def main(check_only=False):
    df = seller_metrics()
    print(json.dumps(df.attrs["audit"], ensure_ascii=False, indent=2))
    if check_only:
        return df.attrs["audit"]
    print(f"Vendedores con >= {MIN_PEDIDOS} pedidos: {len(df):,}")
    X = df[FEATURES]
    for k in range(2, 8):
        p = Pipeline([("scaler", StandardScaler()), ("km", KMeans(k, n_init=10, random_state=42))]).fit(X)
        print(f"k={k}: silhouette={silhouette_score(p[0].transform(X), p[1].labels_):.3f}")

    k = 4  # elegir según el gráfico del codo / silhouette
    pipe = Pipeline([("scaler", StandardScaler()), ("km", KMeans(k, n_init=10, random_state=42))]).fit(X)
    df["cluster"] = pipe[1].labels_
    etiquetas = label_clusters(df.groupby("cluster")[FEATURES].mean())
    df["etiqueta"] = df["cluster"].map(etiquetas)
    print(df.groupby("etiqueta")[FEATURES].mean().round(2))

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "etiquetas": etiquetas}, MODELS / "seller_kmeans.joblib")
    df.reset_index().to_csv(MODELS / "seller_clusters.csv", index=False)
    print("Guardado en", MODELS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true", help="Solo preparación; no ajustar K-Means")
    main(check_only=parser.parse_args().check_data)
