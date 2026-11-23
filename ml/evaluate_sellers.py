"""Evaluación comparativa de clustering de vendedores (Olist)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from train_sellers import FEATURES, MIN_PEDIDOS, build_pipeline, label_clusters, seller_metrics

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def evaluate() -> dict:
    df = seller_metrics()
    X = df[FEATURES]
    base_pipe = build_pipeline(k=4, random_state=42).fit(X)
    X_trans = base_pipe["prep"].transform(X)

    # 1. Barrido de k = 2..8 (Codo + métricas internas)
    k_sweep = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X_trans)
        labels = km.labels_
        k_sweep.append({
            "k": k,
            "inertia": round(float(km.inertia_), 2),
            "silhouette": round(float(silhouette_score(X_trans, labels)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(X_trans, labels)), 2),
            "davies_bouldin": round(float(davies_bouldin_score(X_trans, labels)), 4),
        })

    # 2. Comparación entre algoritmos con k=4
    k_opt = 4
    km_labels = base_pipe["km"].labels_
    agg_labels = AgglomerativeClustering(n_clusters=k_opt, linkage="ward").fit_predict(X_trans)
    gmm_labels = GaussianMixture(n_components=k_opt, random_state=42).fit_predict(X_trans)

    algorithms = []
    for name, lbls in [
        ("K-Means (seleccionado)", km_labels),
        ("Agglomerative (Ward)", agg_labels),
        ("Gaussian Mixture (GMM)", gmm_labels),
    ]:
        algorithms.append({
            "algoritmo": name,
            "silhouette": round(float(silhouette_score(X_trans, lbls)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(X_trans, lbls)), 2),
            "davies_bouldin": round(float(davies_bouldin_score(X_trans, lbls)), 4),
        })

    # 3. Estabilidad de K-Means (ARI respecto a seed=42)
    aris = []
    for seed in [0, 7, 21, 99, 123]:
        other = KMeans(n_clusters=k_opt, n_init=20, random_state=seed).fit(X_trans).labels_
        aris.append(float(adjusted_rand_score(km_labels, other)))

    # 4. Perfil de los 4 segmentos finales
    df_eval = df.copy()
    df_eval["cluster"] = km_labels
    etiquetas = label_clusters(df_eval.groupby("cluster")[FEATURES].mean())
    df_eval["etiqueta"] = df_eval["cluster"].map(etiquetas)
    profiles = (
        df_eval.groupby(["cluster", "etiqueta"])[FEATURES]
        .mean()
        .round(2)
        .assign(n_vendedores=df_eval.groupby(["cluster", "etiqueta"]).size())
        .reset_index()
        .to_dict(orient="records")
    )

    report = {
        "audit": df.attrs["audit"],
        "min_pedidos": MIN_PEDIDOS,
        "features": FEATURES,
        "k_seleccionado": k_opt,
        "estabilidad_ari_medio": round(float(np.mean(aris)), 4),
        "barrido_k": k_sweep,
        "comparacion_algoritmos": algorithms,
        "perfiles_clusters": profiles,
    }

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "evaluacion_sellers_metricas.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_lines = [
        "# Evaluación del Módulo de Clustering de Vendedores (Olist)",
        "",
        f"- **Vendedores elegibles (≥ {MIN_PEDIDOS} pedidos):** {len(df):,}",
        f"- **Variables utilizadas:** `{', '.join(FEATURES)}` (con transformación `log1p` en `n_pedidos` y `facturacion`).",
        f"- **Estabilidad del modelo (k={k_opt}, ARI promedio en 5 semillas):** `{report['estabilidad_ari_medio']}`",
        "",
        "## 1. Selección de número de clústeres (k)",
        "| k | Inercia | Silhouette (↑) | Calinski-Harabasz (↑) | Davies-Bouldin (↓) |",
        "|---:|---:|---:|---:|---:|",
        *[
            f"| {r['k']} | {r['inertia']} | {r['silhouette']} | {r['calinski_harabasz']} | {r['davies_bouldin']} |"
            for r in k_sweep
        ],
        "",
        "## 2. Comparación de algoritmos (k=4)",
        "| Algoritmo | Silhouette (↑) | Calinski-Harabasz (↑) | Davies-Bouldin (↓) |",
        "|---|---:|---:|---:|",
        *[
            f"| {r['algoritmo']} | {r['silhouette']} | {r['calinski_harabasz']} | {r['davies_bouldin']} |"
            for r in algorithms
        ],
        "",
        "## 3. Perfiles de los segmentos obtenidos",
        "| Clúster | Etiqueta | Vendedores | Pedidos medios | Facturación (BRL) | Nota media | Tasa retraso | Entrega (días) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {p['cluster']} | {p['etiqueta']} | {p['n_vendedores']} | {p['n_pedidos']} | {p['facturacion']} | {p['nota_media']} | {p['tasa_retraso']} | {p['dias_entrega_medio']} |"
            for p in profiles
        ],
    ]
    (DOCS / "evaluacion_sellers.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("Reportes generados en docs/evaluacion_sellers.md y docs/evaluacion_sellers_metricas.json")
    return report


if __name__ == "__main__":
    evaluate()