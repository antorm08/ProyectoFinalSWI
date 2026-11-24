"""4. Clustering de vendedores según su desempeño (K-Means).

Salidas: backend/app/models/seller_kmeans.joblib y seller_clusters.csv
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

if __package__:
    from .common import MODELS, checked_join, delivered_orders, load, require_unique
else:
    from common import MODELS, checked_join, delivered_orders, load, require_unique

FEATURES = ["n_pedidos", "facturacion", "nota_media", "tasa_retraso", "dias_entrega_medio"]
LOG_FEATURES = ["n_pedidos", "facturacion"]
LINEAR_FEATURES = ["nota_media", "tasa_retraso", "dias_entrega_medio"]
MIN_PEDIDOS = 5


def build_pipeline(k: int, random_state: int = 42) -> Pipeline:
    """Crea el pipeline con transformación log1p para variables sesgadas y escalamiento."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "log_scale",
                Pipeline([
                    ("log", FunctionTransformer(np.log1p, validate=False)),
                    ("scaler", StandardScaler()),
                ]),
                LOG_FEATURES,
            ),
            ("linear_scale", StandardScaler(), LINEAR_FEATURES),
        ]
    )
    return Pipeline([
        ("prep", preprocessor),
        ("km", KMeans(n_clusters=k, n_init=20, random_state=random_state)),
    ])


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
    checked_join(items[["order_id"]], source[["order_id"]], "order_id")
    checked_join(reviews[["order_id"]], source[["order_id"]], "order_id")
    items["price"] = pd.to_numeric(items["price"], errors="raise")
    if items["price"].isna().any() or not np.isfinite(items["price"]).all() or items["price"].lt(0).any():
        raise ValueError("Precio faltante, negativo o infinito")
    reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="raise")
    if (~reviews["review_score"].between(1, 5) | reviews["review_score"].mod(1).ne(0)).any():
        raise ValueError("Nota de reseña fuera de 1–5 o no entera")
    selected = items.loc[items["order_id"].isin(eligible["order_id"])]
    pairs = selected.groupby(["seller_id", "order_id"], as_index=False).agg(
        facturacion=("price", "sum"), n_articulos=("order_item_id", "size")
    )
    pairs = checked_join(pairs, eligible[["order_id", "retrasado", "dias_entrega"]], "order_id")
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
    missing = metrics[FEATURES].isna().any(axis=1)
    result = metrics.loc[minimum & ~missing].copy()
    result.attrs["audit"] = {
        **pairs.attrs["audit"], "sellers_with_eligible_orders": len(metrics),
        "sellers_below_minimum": int((~minimum).sum()),
        "sellers_missing_metrics_after_minimum": int((minimum & missing).sum()),
        "eligible_sellers": len(result),
    }
    return result


def label_clusters(centers: pd.DataFrame) -> dict[int, str]:
    """Asigna etiquetas únicas y accionables según los centroides en escala original."""
    remaining = set(centers.index)
    labels: dict[int, str] = {}

    c_delay = int(centers.loc[sorted(remaining), "tasa_retraso"].idxmax())
    labels[c_delay] = "Logística problemática"
    remaining.remove(c_delay)

    if remaining:
        c_top = int(centers.loc[sorted(remaining), "facturacion"].idxmax())
        labels[c_top] = "Vendedores estrella (alto volumen)"
        remaining.remove(c_top)

    if len(remaining) > 1:
        c_low = int(centers.loc[sorted(remaining), "nota_media"].idxmin())
        labels[c_low] = "Baja satisfacción"
        remaining.remove(c_low)

    estables = [
        c for c in sorted(remaining)
        if centers.loc[c, "tasa_retraso"] <= centers["tasa_retraso"].median()
        and centers.loc[c, "nota_media"] >= centers["nota_media"].median()
    ]
    if estables:
        c_stable = int(centers.loc[estables, "tasa_retraso"].idxmin())
        labels[c_stable] = "Vendedores estables"
        remaining.remove(c_stable)
    for c in sorted(remaining):
        labels[int(c)] = f"Perfil intermedio (grupo {c + 1})"

    return labels


def main(check_only=False, *, models_dir=MODELS, metrics_path=None):
    df = seller_metrics()
    print(json.dumps(df.attrs["audit"], ensure_ascii=False, indent=2))
    if check_only:
        return df.attrs["audit"]
    models_dir = Path(models_dir)
    model_path, table_path = models_dir / "seller_kmeans.joblib", models_dir / "seller_clusters.csv"
    for target in (model_path, table_path, metrics_path):
        if target is not None and Path(target).exists():
            raise FileExistsError(f"Respalda los resultados anteriores o elige otra ruta: {target}")
    print(f"Vendedores con >= {MIN_PEDIDOS} pedidos: {len(df):,}")
    X = df[FEATURES]
    candidates = {}
    for k in range(2, min(8, len(X) - 1) + 1):
        p = build_pipeline(k).fit(X)
        score = float(silhouette_score(p["prep"].transform(X), p["km"].labels_))
        candidates[k] = (score, p)
        print(f"k={k}: silhouette={score:.4f}")

    k = max(candidates, key=lambda n: (candidates[n][0], -n))
    score, pipe = candidates[k]
    df["cluster"] = pipe["km"].labels_
    etiquetas = label_clusters(df.groupby("cluster")[FEATURES].mean())
    df["etiqueta"] = df["cluster"].map(etiquetas)
    profiles = df.groupby(["cluster", "etiqueta"])[FEATURES].agg("mean")
    profiles["n_vendedores"] = df.groupby(["cluster", "etiqueta"]).size()
    print(profiles.round(4))

    models_dir.mkdir(parents=True, exist_ok=True)
    with model_path.open("xb") as file:
        joblib.dump({"pipeline": pipe, "etiquetas": etiquetas}, file)
    df.reset_index().to_csv(table_path, index=False, mode="x")
    np.testing.assert_array_equal(joblib.load(model_path)["pipeline"].predict(X), df["cluster"])
    report = {
        "audit": df.attrs["audit"], "features": FEATURES, "k_seleccionado": k,
        "criterio": "Máximo Silhouette sobre las mismas variables escaladas; empate por k menor.",
        "silhouette": score, "barrido_k": {n: value for n, (value, _) in candidates.items()},
        "perfiles_clusters": profiles.reset_index().to_dict(orient="records"),
        "artefactos_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (model_path, table_path)},
    }
    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
    print("Guardado en", models_dir)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true", help="Solo preparación; no ajustar K-Means")
    parser.add_argument("--models-dir", type=Path, default=MODELS)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    main(check_only=args.check_data, models_dir=args.models_dir, metrics_path=args.metrics)
