# Evaluación del Módulo de Clustering de Vendedores (Olist)

> Este informe conserva el experimento anterior con k=4. La demo sirve el ajuste de **k=3**,
> seleccionado por Silhouette; ver [métricas, perfiles y artefactos activos](activacion_vendedores_metricas.json).

- **Vendedores elegibles (≥ 5 pedidos):** 1,766
- **Variables utilizadas:** `n_pedidos, facturacion, nota_media, tasa_retraso, dias_entrega_medio` (con transformación `log1p` en `n_pedidos` y `facturacion`).
- **Estabilidad del modelo (k=4, ARI promedio en 5 semillas):** `0.9491`

## 1. Selección de número de clústeres (k)
| k | Inercia | Silhouette (↑) | Calinski-Harabasz (↑) | Davies-Bouldin (↓) |
|---:|---:|---:|---:|---:|
| 2 | 6416.34 | 0.2664 | 663.57 | 1.4355 |
| 3 | 4689.26 | 0.3053 | 778.39 | 1.1225 |
| 4 | 4095.72 | 0.2515 | 678.91 | 1.3713 |
| 5 | 3674.99 | 0.221 | 617.55 | 1.3341 |
| 6 | 3327.44 | 0.2268 | 582.1 | 1.3012 |
| 7 | 3102.41 | 0.2218 | 541.24 | 1.3037 |
| 8 | 2900.55 | 0.2179 | 513.4 | 1.3041 |

## 2. Comparación de algoritmos (k=4)
| Algoritmo | Silhouette (↑) | Calinski-Harabasz (↑) | Davies-Bouldin (↓) |
|---|---:|---:|---:|
| K-Means (seleccionado) | 0.2515 | 678.91 | 1.3713 |
| Agglomerative (Ward) | 0.1672 | 545.82 | 1.298 |
| Gaussian Mixture (GMM) | 0.2001 | 485.17 | 1.6352 |

## 3. Perfiles de los segmentos obtenidos
| Clúster | Etiqueta | Vendedores | Pedidos medios | Facturación (BRL) | Nota media | Tasa retraso | Entrega (días) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Vendedores estrella (alto volumen) | 504 | 151.32 | 19574.51 | 4.19 | 0.06 | 12.15 |
| 1 | Vendedores estables | 584 | 12.46 | 1426.55 | 4.52 | 0.01 | 9.01 |
| 2 | Logística problemática | 151 | 17.7 | 3620.59 | 3.5 | 0.24 | 20.11 |
| 3 | Baja satisfacción | 527 | 17.47 | 2749.47 | 4.0 | 0.08 | 12.98 |
