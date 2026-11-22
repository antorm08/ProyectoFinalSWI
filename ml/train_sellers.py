"""4. Clustering de vendedores según su desempeño (K-Means).

Salidas: backend/app/models/seller_kmeans.joblib y seller_clusters.csv
"""
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import MODELS, delivered_orders, load

FEATURES = ["n_pedidos", "facturacion", "nota_media", "tasa_retraso", "dias_entrega_medio"]
MIN_PEDIDOS = 5


def seller_metrics():
    orders = delivered_orders()[["order_id", "retrasado", "dias_entrega"]]
    reviews = load("order_reviews").groupby("order_id")["review_score"].mean().rename("nota")
    items = load("order_items").merge(orders, on="order_id").merge(reviews, on="order_id", how="left")
    m = items.groupby("seller_id").agg(
        n_pedidos=("order_id", "nunique"), facturacion=("price", "sum"), nota_media=("nota", "mean"),
        tasa_retraso=("retrasado", "mean"), dias_entrega_medio=("dias_entrega", "mean"))
    return m[m["n_pedidos"] >= MIN_PEDIDOS].dropna()


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


def main():
    df = seller_metrics()
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

    joblib.dump({"pipeline": pipe, "etiquetas": etiquetas}, MODELS / "seller_kmeans.joblib")
    df.reset_index().to_csv(MODELS / "seller_clusters.csv", index=False)
    print("Guardado en", MODELS)


if __name__ == "__main__":
    main()
