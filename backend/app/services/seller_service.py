"""Segmentación de vendedores por desempeño (clustering K-Means)."""
import joblib
import pandas as pd

from app.core.config import SELLER_MODEL, SELLER_TABLE
from app.schemas import SellerClusterOutput, SellerMetrics

FEATURES = ["n_pedidos", "facturacion", "nota_media", "tasa_retraso", "dias_entrega_medio"]

_bundle = None


def _get_bundle():
    """El bundle contiene {'pipeline': Pipeline(scaler+kmeans), 'etiquetas': {cluster: nombre}}."""
    global _bundle
    if _bundle is None and SELLER_MODEL.exists():
        _bundle = joblib.load(SELLER_MODEL)
    return _bundle


def summary() -> dict:
    if not SELLER_TABLE.is_file() or not SELLER_MODEL.is_file():
        return {"modelo_entrenado": False, "clusters": [], "vendedores": []}
    df = pd.read_csv(SELLER_TABLE)
    g = df.groupby(["cluster", "etiqueta"])
    clusters = (g[FEATURES].mean().round(2).assign(n_vendedores=g.size())
                  .reset_index().to_dict(orient="records"))
    vendedores = df.sort_values("facturacion", ascending=False).head(300).round(3).to_dict(orient="records")
    return {"modelo_entrenado": True, "clusters": clusters, "vendedores": vendedores}


def assign(m: SellerMetrics) -> SellerClusterOutput:
    bundle = _get_bundle()
    if bundle is None:
        return SellerClusterOutput(cluster=-1, etiqueta="(modelo no entrenado)", modelo_entrenado=False)
    c = int(bundle["pipeline"].predict(pd.DataFrame([m.model_dump()])[FEATURES])[0])
    return SellerClusterOutput(cluster=c, etiqueta=bundle["etiquetas"].get(c, f"Grupo {c}"), modelo_entrenado=True)
