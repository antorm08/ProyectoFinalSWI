# Validación integral de Olist para CommerceAI

**Fecha:** 22 de noviembre de 2026.  
**Ubicación:** `data/olist/` de la copia principal.  
**Dictamen:** los seis archivos son legibles y sus relaciones son consistentes. Hay datos suficientes para los componentes basados en Olist, pero **no se recomienda entrenar retrasos ni vendedores sin resolver las observaciones metodológicas de este informe**.

> Este informe describe el estado anterior a las correcciones. El estado corregido, sus políticas, conteos y pruebas se documentan en [preparacion_olist.md](preparacion_olist.md). Las estadísticas originales se conservan como referencia.

## 1. Alcance y metodología

- Lectura completa de los seis CSV con UTF-8, `pandas.read_csv` y error ante filas mal formadas; sin muestreo para perfiles, duplicados, relaciones o conteos.
- Comprobación de esquema, tipos inferidos, nulos, cadenas vacías, claves, fechas, dominios numéricos y cardinalidades.
- Reconstrucción en memoria de las transformaciones de `ml/common.py`, `ml/train_delay.py`, `ml/train_demand.py` y `ml/train_sellers.py`, **sin llamar a `fit`, entrenar modelos ni escribir artefactos**.
- Reproducción del split aleatorio de retrasos con `test_size=0.2`, `stratify=retrasado`, `random_state=42`.
- Comparación diagnóstica de métricas de vendedores por artículo frente a métricas por pareja vendedor-pedido. No se sustituyeron las métricas existentes.
- Prueba de las funciones reales de TextRank sobre 11 agrupaciones de reseñas reales, incluida la de mayor número de oraciones elegibles.
- Prueba del cargador real del backend y de los endpoints de estado, demanda y resumen por vendedor.
- SHA-256 de los CSV antes y después del análisis: **sin cambios**. No se modificaron CSV, aplicación, algoritmos ni modelos.

Entorno utilizado: Python 3.12.10, pandas 3.1.1, NumPy 2.5.5, scikit-learn 1.9.1 y statsmodels 0.15.0. Los conteos son resultados de esta ejecución, no cifras tomadas de documentación externa.

Los archivos fueron aportados como la versión oficial de Olist. El esquema y los volúmenes son compatibles con ella; los hashes identifican esta copia local, **no certifican su autenticidad frente a una fuente externa**.

## 2. Inventario y carga

| Archivo | Filas | Columnas | Clave comprobada | Filas completas duplicadas | Repeticiones adicionales de clave |
|---|---:|---:|---|---:|---:|
| `olist_orders_dataset.csv` | 99.441 | 8 | `order_id` | 0 | 0 |
| `olist_customers_dataset.csv` | 99.441 | 5 | `customer_id` | 0 | 0 |
| `olist_order_items_dataset.csv` | 112.650 | 7 | `order_id, order_item_id` | 0 | 0 |
| `olist_products_dataset.csv` | 32.951 | 9 | `product_id` | 0 | 0 |
| `olist_sellers_dataset.csv` | 3.095 | 4 | `seller_id` | 0 | 0 |
| `olist_order_reviews_dataset.csv` | 99.224 | 7 | `review_id` no es único | 0 | 814 |

**Resultado:** 6/6 archivos cargados, sin columnas esperadas ausentes ni columnas adicionales. Las 40 columnas se detallan abajo. Todas las claves de texto comprobadas son identificadores hexadecimales de 32 caracteres, sin nulos ni cadenas vacías.

En reseñas existen **98.410 `review_id` distintos**. Hay 789 identificadores reutilizados, que afectan a 1.603 filas; por eso sobran 814 filas si se trata incorrectamente `review_id` como clave única. La pareja **`review_id, order_id` sí es única** en esta copia.

### 2.1 Columnas, tipos y nulos

`str` es el tipo inferido en este entorno. Las fechas llegan como texto; todas las fechas no nulas se convierten correctamente a `datetime64[us]`, sin zona horaria. No hubo conversiones fallidas.

#### Pedidos

| Columna | Tipo inferido | Nulos |
|---|---|---:|
| `order_id` | str | 0 |
| `customer_id` | str | 0 |
| `order_status` | str | 0 |
| `order_purchase_timestamp` | str, convertible a fecha | 0 |
| `order_approved_at` | str, convertible a fecha | 160 |
| `order_delivered_carrier_date` | str, convertible a fecha | 1.783 |
| `order_delivered_customer_date` | str, convertible a fecha | 2.965 |
| `order_estimated_delivery_date` | str, convertible a fecha | 0 |

#### Clientes

| Columna | Tipo inferido | Nulos |
|---|---|---:|
| `customer_id` | str | 0 |
| `customer_unique_id` | str | 0 |
| `customer_zip_code_prefix` | int64 | 0 |
| `customer_city` | str | 0 |
| `customer_state` | str | 0 |

#### Ítems

| Columna | Tipo inferido | Nulos |
|---|---|---:|
| `order_id` | str | 0 |
| `order_item_id` | int64 | 0 |
| `product_id` | str | 0 |
| `seller_id` | str | 0 |
| `shipping_limit_date` | str, convertible a fecha | 0 |
| `price` | float64 | 0 |
| `freight_value` | float64 | 0 |

#### Productos

| Columna | Tipo inferido | Nulos |
|---|---|---:|
| `product_id` | str | 0 |
| `product_category_name` | str | 610 |
| `product_name_lenght` | float64 | 610 |
| `product_description_lenght` | float64 | 610 |
| `product_photos_qty` | float64 | 610 |
| `product_weight_g` | float64 | 2 |
| `product_length_cm` | float64 | 2 |
| `product_height_cm` | float64 | 2 |
| `product_width_cm` | float64 | 2 |

Los nombres `*_lenght` conservan la grafía del CSV. Los conteos y dimensiones tienen tipo flotante por la presencia de nulos; no son necesariamente magnitudes fraccionarias.

#### Vendedores

| Columna | Tipo inferido | Nulos |
|---|---|---:|
| `seller_id` | str | 0 |
| `seller_zip_code_prefix` | int64 | 0 |
| `seller_city` | str | 0 |
| `seller_state` | str | 0 |

#### Reseñas

| Columna | Tipo inferido | Nulos |
|---|---|---:|
| `review_id` | str | 0 |
| `order_id` | str | 0 |
| `review_score` | int64 | 0 |
| `review_comment_title` | str | 87.656 |
| `review_comment_message` | str | 58.247 |
| `review_creation_date` | str, convertible a fecha | 0 |
| `review_answer_timestamp` | str, convertible a fecha | 0 |

Además de los nulos, hay **2 títulos y 27 mensajes formados solo por espacios o caracteres vacíos tras `strip()`**. No se encontraron cadenas vacías en las demás columnas textuales perfiladas.

### 2.2 Interpretación de tipos

- Los prefijos postales son códigos, no variables continuas. `read_csv` los convierte a enteros y pierde los ceros iniciales; si se muestran o usan como categorías, tratarlos como texto de cinco posiciones. Ningún prefijo está fuera de 0–99.999 ni es fraccionario.
- Clientes: 27 estados válidos de Brasil; vendedores: 23. No hay siglas inválidas.
- Hay 96.096 `customer_unique_id` distintos y 2.997 compradores asociados a más de un `customer_id`.
- Cada `customer_id` está asociado a un único pedido en esta copia. Para detectar compradores repetidos o separar grupos debe usarse **`customer_unique_id`**, no `customer_id`.

## 3. Integridad referencial y cardinalidades

| Relación | Filas hijas | Claves nulas | Filas huérfanas |
|---|---:|---:|---:|
| `orders.customer_id → customers.customer_id` | 99.441 | 0 | 0 |
| `order_items.order_id → orders.order_id` | 112.650 | 0 | 0 |
| `order_items.product_id → products.product_id` | 112.650 | 0 | 0 |
| `order_items.seller_id → sellers.seller_id` | 112.650 | 0 | 0 |
| `order_reviews.order_id → orders.order_id` | 99.224 | 0 | 0 |

- Los 32.951 productos y 3.095 vendedores aparecen en ítems; todos los clientes están referenciados por pedidos.
- Hay **98.666 pedidos con ítems** y **775 sin ítems**. Estos últimos son: 603 `unavailable`, 164 `canceled`, 5 `created`, 2 `invoiced` y 1 `shipped`. No hay pedidos `delivered` sin ítems.
- Hay **98.673 pedidos con reseñas** y **768 sin reseñas**: 646 `delivered`, 75 `shipped`, 20 `canceled`, 14 `unavailable`, 6 `processing`, 5 `invoiced` y 2 `created`.
- Ítems por pedido: mínimo 1, mediana 1 y máximo 21. La numeración de `order_item_id` comienza en 1 y es consecutiva en todos los pedidos comprobados.
- **1.278 pedidos tienen varios vendedores**: 1.219 tienen 2; 54 tienen 3; 3 tienen 4; 2 tienen 5.
- **547 pedidos tienen varias reseñas**: 543 tienen 2 y 4 tienen 3. Un join directo de ítems con reseñas multiplicaría filas; primero debe definirse cómo agregar las reseñas por pedido.
- Los 789 `review_id` reutilizados se vinculan a varios pedidos del **mismo comprador** y no presentan conflictos en nota, mensaje no nulo, fecha de creación o respuesta. No son filas completas duplicadas. No deben eliminarse globalmente por `review_id` sin definir la unidad de análisis.

Las relaciones son utilizables; **ausencia de hijos no equivale a clave huérfana**, y las relaciones de reseñas no son uno-a-uno.

## 4. Fechas, precios y estados

### 4.1 Estados de pedidos

| Estado | Pedidos |
|---|---:|
| delivered | 96.478 |
| shipped | 1.107 |
| canceled | 625 |
| unavailable | 609 |
| invoiced | 314 |
| processing | 301 |
| created | 5 |
| approved | 2 |

Los nulos de entrega son esperables en estados todavía no entregados. No deben imputarse como entregas puntuales.

Casos que requieren revisión:

- 8 pedidos `delivered` sin fecha de entrega al cliente.
- 14 `delivered` sin aprobación y 2 sin entrega al transportista; estos conjuntos pueden solaparse.
- 6 `canceled` con fecha de entrega al cliente.
- Los 1.107 `shipped` no tienen entrega al cliente. No es correcto tratarlos como negativos de retraso ni concluir su resultado final.

### 4.2 Orden temporal

Se compara únicamente cuando ambas fechas están presentes. Los conteos pueden solaparse y no deben sumarse.

| Comprobación | Pares comparables | Casos |
|---|---:|---:|
| Aprobación anterior a compra | 99.281 | 0 |
| Entrega al transportista anterior a compra | 97.658 | 166 |
| Entrega al transportista anterior a aprobación | 97.644 | 1.359 |
| Entrega al cliente anterior a compra | 96.476 | 0 |
| Entrega al cliente anterior a aprobación | 96.462 | 61 |
| Entrega al cliente anterior a transportista | 96.475 | 23 |
| Fecha estimada anterior a compra | 99.441 | 0 |
| Límite de envío anterior a compra, por ítem | 112.650 | 0 |
| Límite de envío anterior a aprobación, por ítem | 112.635 | 127 |
| Creación de reseña anterior al timestamp de compra | 99.224 | 74 |
| Respuesta de reseña anterior a creación | 99.224 | 0 |

En los pedidos entregados con objetivo calculable, **1.373 pedidos** presentan al menos una de las anomalías de transportista/aprobación/entrega anteriores. No implica que su etiqueta de retraso sea necesariamente incorrecta; requiere una política de calidad.

Matices importantes:

- En 679 de los 1.359 casos transportista-antes-aprobación ambos eventos son del mismo día; puede influir la captura de horas, pero no explica todos los casos.
- 64 reseñas se crean en un día anterior a la compra, y 63 respuestas también son anteriores al timestamp de compra. Revisar asociaciones y cronología, especialmente los identificadores compartidos.
- 8.320 reseñas se crean antes del timestamp de entrega y 5.127 antes de su día calendario. **No es automáticamente un error**: una encuesta puede emitirse antes de una entrega retrasada; además, la creación de reseña no tiene hora distinta de medianoche.
- `shipping_limit_date` llega hasta **2020-04-09**: hay 4 ítems de 3 pedidos con límite de envío a más de 365 días de la compra. En total, 372 ítems tienen límite posterior a la entrega estimada. Son candidatos a revisión; el límite de envío no se usa actualmente como predictor.
- La duración máxima observada entre compra y entrega es 209,63 días. No se detectaron duraciones negativas de compra a entrega entre los pedidos utilizados por los scripts.

### 4.3 Precios y atributos de producto

| Campo | Nulos | Ceros | Negativos | Mínimo | Mediana | Máximo |
|---|---:|---:|---:|---:|---:|---:|
| Precio por ítem, BRL | 0 | 0 | 0 | 0,85 | 74,99 | 6.735,00 |
| Flete por ítem, BRL | 0 | 383 | 0 | 0,00 | 16,26 | 409,68 |
| Peso de producto, g | 2 | 4 | 0 | 0 | 700 | 40.425 |
| Largo, cm | 2 | 0 | 0 | 7 | 25 | 105 |
| Alto, cm | 2 | 0 | 0 | 2 | 13 | 105 |
| Ancho, cm | 2 | 0 | 0 | 6 | 20 | 118 |

No se encontraron infinitos en las columnas numéricas. Todas las notas de reseña son enteros entre 1 y 5.

Los fletes de cero pueden ser válidos. Los importes o pesos altos son **valores extremos**, no errores demostrados; no corresponde eliminarlos automáticamente. Los 4 productos de peso cero sí requieren una decisión de tratamiento.

La suma de `price` es valor de artículos, no cobros netos ni facturación contable confirmada: estos seis CSV no incluyen pagos, devoluciones monetarias ni costos.

## 5. Viabilidad por componente

### 5.1 Predicción de retrasos

Archivos: `ml/common.py:19-30`, `ml/train_delay.py:21-54`; consumo: `backend/app/services/delay_service.py`.

| Paso del procesamiento actual | Pedidos |
|---|---:|
| Estado delivered | 96.478 |
| Con fechas para calcular objetivo | 96.470 |
| Tras joins y agregación de variables | 96.470 |
| Tras `dropna` de variables y objetivo | 96.470 |
| Etiqueta puntual, comparación por timestamp | 88.644 |
| Etiqueta retrasado, comparación por timestamp | 7.826 |

**Tasa positiva actual: 8,11 %.** Las 10 variables actuales son numéricamente finitas cuando corresponde y están disponibles tras los joins.

Problemas antes de entrenar:

1. **Semántica del objetivo:** las 99.441 fechas estimadas están a medianoche. El objetivo compara el timestamp real con esa medianoche, de modo que **1.292 entregas del mismo día prometido se etiquetan como tardías**. Si el compromiso se interpreta por día calendario, los retrasos pasan de 7.826 a **6.534**, es decir, **6,77 %**. La definición debe aprobarse antes de cambiarla; no se cambió ninguna etiqueta persistida.
2. **Peso ausente convertido en cero:** los 2 productos sin peso aparecen en 16 pedidos elegibles. `groupby.sum()` devuelve cero cuando todos los pesos son nulos, de modo que `dropna()` no los elimina. Hay **22 filas de entrenamiento con peso agregado cero**, incluyendo los casos con pesos explícitamente cero. Conviene preservar el faltante y tratarlo explícitamente.
3. **Pedidos multivendedor:** `estado_vendedor=("seller_state","first")` conserva un único estado. Hay **509 pedidos elegibles con vendedores de estados distintos**. Elegir el primero depende del orden de filas y no representa el envío completo. Debe definirse una política determinista, sin cambiar silenciosamente el contrato de entrada.
4. **Sesgo de población:** se entrena solo con pedidos finalmente entregados. Es un modelo de retraso condicionado a esa población, no de cancelación o riesgo general de cualquier pedido abierto.
5. **Granularidad de días:** `.dt.days` trunca duraciones. Los plazos prometidos usados van de 2 a 155 días; aclarar cómo deben introducirse esos días en el formulario para mantener consistencia con el entrenamiento.

**Dictamen:** volumen suficiente y preparación ejecutable; **entrenamiento condicionado** a definir el objetivo, política de nulos/multivendedor y validación temporal.

### 5.2 Pronóstico de demanda

Archivos: `ml/train_demand.py:13-33`, `backend/app/services/demand_service.py:21-51`.

- Compras disponibles: **2016-09-04 21:15:19 a 2018-10-17 17:30:18**.
- Serie completa reindexada: 774 días, 140 días sin registros.
- Ventana actual del script: **2017-01-01 a 2018-08-15**, 592 días y **96.912 pedidos**, el 97,46 % del archivo.
- Los 4 días con cero dentro de esa ventana son 2017-01-01, 02, 03 y 04. Debe distinguirse ausencia de registros de demanda verdaderamente nula.
- Mediana diaria: 151,5; promedio: 163,70; máximo: **1.176 pedidos el 2017-11-24**. Los siguientes máximos son 499 el 25, 403 el 27, 391 el 26 y 380 el 28 de noviembre de 2017. No deben descartarse como errores sin contexto comercial.
- La ventana contiene 94.079 `delivered` y **2.833 pedidos en otros estados**: 1.085 `shipped`, 601 `unavailable`, 545 `canceled`, 299 `processing`, 296 `invoiced`, 5 `created` y 2 `approved`.

La variable objetivo actual es **pedidos creados por día, de cualquier estado**, no pedidos entregados, unidades ni ventas. Es válida como definición de demanda si se comunica así; filtrar por estado final puede introducir sesgo de supervivencia.

Validación actual:

| Conjunto | Días | Período |
|---|---:|---|
| Entrenamiento | 562 | 2017-01-01 a 2018-07-16 |
| Prueba | 30 | 2018-07-17 a 2018-08-15 |

La prueba tiene entre 176 y 372 pedidos diarios, promedio 267,37 y ningún día de cero. El split respeta el orden temporal; no se detectó uso del tramo de prueba para ajustar el primer modelo en el código.

Limitaciones:

- El recorte excluye **2.529 pedidos**. Después del 15 de agosto todavía hay 2.132 entregados, 54 cancelados, 13 enviados y 1 no disponible. Documentar por qué esa fecha es el corte elegido; no asumir que todo lo excluido es incompleto.
- Septiembre y octubre de 2018 solo contienen 16 y 4 compras, respectivamente. Pronosticar sobre toda la cola sin estudiar la cobertura produciría una caída artificial.
- SARIMA usa estacionalidad de 7 días. Hay unas 84 semanas, suficientes para evaluarla; no hay dos ciclos anuales completos dentro de la ventana.
- Se evalúa un único holdout y no se compara con un baseline semanal ni con varios cortes temporales.
- El MAPE divide por `test.clip(lower=1)`: en esta prueba no afecta porque no hay ceros, pero con otros períodos deja de ser el MAPE convencional para días nulos.
- El ajuste final con toda la serie después de evaluar es normal para generar el artefacto; sus predicciones sobre el mismo holdout no deben presentarse como evaluación fuera de muestra.

**Dictamen:** **apto para un entrenamiento exploratorio** con el objetivo y recorte documentados. No se entrenó y no hay métricas de precisión nuevas.

### 5.3 Clustering de vendedores

Archivo: `ml/train_sellers.py:17-61`.

| Paso actual | Vendedores |
|---|---:|
| Registrados y con ítems | 3.095 |
| Con ítems de pedidos delivered y fechas calculables | 2.970 |
| Excluidos por menos de 5 pedidos elegibles | 1.204 |
| Elegibles antes de `dropna` | 1.766 |
| Elegibles después de `dropna` | 1.766 |

Los **1.766 vendedores** representan el 57,06 % de los registrados; tienen 1.766 vectores de características distintos. Hay suficientes muestras para explorar los `k=2…7` del script. No se ejecutó K-Means ni silhouette.

Problemas de agregación:

- El procesamiento contiene **110.189 filas de artículos** pero solo **97.811 parejas vendedor-pedido**: hay 12.378 filas adicionales por pedidos con varios artículos, no duplicados inválidos.
- `n_pedidos` usa pedidos distintos y `facturacion` suma precios de artículos; esas agregaciones respetan su unidad.
- En cambio, `nota_media`, `tasa_retraso` y `dias_entrega_medio` calculan medias sobre las filas de artículos. Un pedido de varios artículos tiene más peso que otro de uno.

Comparación diagnóstica entre el cálculo actual y una media por vendedor-pedido, manteniendo los mismos 1.766 vendedores y las mismas etiquetas:

| Métrica | Vendedores con diferencia | Diferencia absoluta máxima |
|---|---:|---:|
| Nota media | 1.138 | 1,3583 puntos |
| Tasa de retraso | 875 | 0,38, equivalente a 38 puntos porcentuales |
| Entrega media | 1.159 | 7,8701 días |

No son cambios propuestos de algoritmo: son diferencias de la **unidad de agregación** que deben resolverse antes de interpretar los grupos como desempeño por pedido.

Otros puntos:

- La tasa de retraso hereda la definición por timestamp de `common.py`.
- Hay 827 filas de artículos elegibles sin nota de reseña. `mean` ignora esos nulos; ningún vendedor con al menos cinco pedidos queda completamente sin métricas en esta copia. Conviene informar cobertura de reseñas, no imputar satisfacción.
- Facturación por vendedor elegible: mediana BRL 2.355,805 y máximo BRL 226.987,93. Pedidos: mediana 17 y máximo 1.819. La asimetría puede influir en K-Means aun con StandardScaler; evaluar estabilidad antes de considerar transformaciones, sin asumir que los extremos son errores.
- Se imprime silhouette para varios `k`, pero el artefacto siempre se genera con **`k=4` fijo**. No hay selección automática ni gráfico del codo en este script.
- Las etiquetas de clusters se asignan por reglas con prioridad y extremos; no garantizan cuatro nombres distintos ni categorías comerciales validadas.
- Las reseñas de un pedido multivendedor son de nivel pedido, no una evaluación inequívoca de cada vendedor.

**Dictamen:** hay datos suficientes, pero **corregir o justificar la ponderación por artículo antes de entrenar e interpretar segmentos**.

### 5.4 Texto para TextRank

Archivo: `backend/app/services/summary_service.py:24-70`.

| Indicador | Resultado |
|---|---:|
| Filas de reseñas | 99.224 |
| Mensajes no nulos | 40.977 |
| Mensajes con contenido tras `strip()` | 40.950 |
| Porcentaje de filas con texto no vacío | 41,27 % |
| Mensajes distintos tras `strip()` | 35.616 |
| Repeticiones adicionales de mensajes | 5.334 |
| Mensajes de al menos 20 caracteres | 34.157 |
| Mensajes con al menos una oración elegible por el código | 35.047 |
| Mensajes con más de 3 oraciones elegibles | 565 |
| Pedidos con mensaje no vacío | 40.809 |
| Vendedores asociados a mensajes no vacíos | 2.555 |
| Vendedores con alguna oración elegible agrupada | 2.471 |
| Vendedores con más de 3 oraciones elegibles agrupadas | 1.482 |
| Vendedores con más de 5 oraciones elegibles agrupadas | 1.154 |

Longitud de mensajes no vacíos: mediana **53 caracteres**, percentil 95 de 192 y máximo 208. El texto individual suele ser breve; **agrupar reseñas por vendedor es más apropiado para demostrar resumen de texto extenso**.

El conteo de oraciones replica la regla actual: al menos tres palabras, eliminación de oraciones repetidas y separación por puntuación o saltos de línea. Los conteos por vendedor usan pedidos únicos por vendedor antes del join; no repiten una reseña por cada artículo.

Comprobación funcional:

- Se resumieron 11 agrupaciones reales con extensión solicitada de 3 oraciones.
- **11/11 devolvieron un resumen no vacío de 3 oraciones, sin excepciones**.
- La mayor agrupación tiene **1.003 oraciones elegibles** y también se procesó correctamente.
- No hubo vocabulario completamente vacío entre las agrupaciones con más de tres oraciones.
- No se ha medido calidad semántica, representatividad ni fidelidad con evaluación humana; ejecutar sin errores no demuestra calidad del resumen.

Limitaciones:

- `dropna()` del servicio no descarta mensajes solo de espacios; 27 mensajes requieren distinguir vacío de texto útil. Puede obtenerse un resultado sin oraciones aunque exista un campo no nulo.
- El título de la reseña no se usa para resumir. No se debe asumir que todos los títulos completan un mensaje.
- Los mensajes repetidos y los `review_id` compartidos pueden sobre-representar opiniones. El servicio deduplica oraciones, no normaliza todas las variantes de texto.
- TF-IDF y similitud usan una matriz densa de tamaño cuadrático. Para 1.003 oraciones hay 1.006.009 celdas: unos 7,68 MiB para una sola matriz float64, sin contar copias y TF-IDF. La muestra actual es viable; no hay límite explícito para entradas arbitrariamente extensas.
- Las reseñas están disponibles después de la compra y no deben convertirse en predictores de retraso al momento de comprar.

**Dictamen:** **apto para TextRank sin entrenamiento**, especialmente mediante agrupación por vendedor, con tratamiento de vacíos y revisión humana de los resúmenes.

## 6. Fugas de información y evaluación

### 6.1 Retrasos: partición actual

Reproduciendo el split existente:

| Indicador | Resultado |
|---|---:|
| Pedidos de entrenamiento | 77.176 |
| Pedidos de prueba | 19.294 |
| `order_id` compartidos | 0 |
| Compradores únicos compartidos | 926 |
| Pedidos de prueba de compradores vistos en entrenamiento | 960 |
| Vendedores compartidos | 1.917 de 2.032 vendedores presentes en prueba |
| Productos compartidos | 6.989 de 11.060 productos presentes en prueba |

Las compras de entrenamiento abarcan 2016-10-03 a 2018-08-29 y las de prueba 2016-09-15 a 2018-08-28. **El entrenamiento contiene futuro respecto de una parte de la prueba.** Este split no estima una operación estrictamente prospectiva.

Matices:

- La lista actual de variables **no incluye directamente entrega real, reseña, nota ni objetivo**. No se encontró una fuga directa del objetivo a `X`.
- Compartir compradores, vendedores o productos **no es automáticamente fuga** cuando representan entidades conocidas en operación. Sí invalida afirmar generalización a entidades nuevas y puede hacer optimista una evaluación de esa población.
- Para una evaluación futura, dividir por compra cronológica. Para evaluar clientes nuevos, agrupar por `customer_unique_id`. El criterio depende de la población de uso y debe documentarse.
- **Un corte cronológico tampoco basta por sí solo:** con corte diagnóstico 2018-07-17, 86.507 pedidos elegibles se compraron antes del corte, pero **1.286 se entregaron en el corte o después**. Sus etiquetas aún no estarían disponibles al entrenar en esa fecha. Debe exigirse madurez del resultado o un período de separación.
- Preprocesamiento del pipeline se ajusta únicamente después del split, lo cual es correcto. La selección de variables, umbrales y cualquier imputación futura deben mantener esa separación.
- `class_weight="balanced"` aborda el desbalance, pero no demuestra calibración de probabilidades. ROC-AUC y F1 no bastan para presentar porcentajes como riesgos calibrados; evaluar también precisión-recall y calibración en datos separados, sin inventar métricas.

### 6.2 Series temporales

El script respeta un holdout final de 30 días. Construir el índice diario a partir de todo el período no implica por sí mismo fuga de valores, pero deben fijarse el recorte y las decisiones de modelo sin optimizar repetidamente contra ese único holdout.

Recomendación: backtesting con varios orígenes, baseline de última semana y un conjunto final reservado. No introducir estados finales o reseñas futuras como información conocida al momento de pronosticar.

### 6.3 Clustering

Agrupar vendedores por todo su desempeño histórico es válido como **análisis retrospectivo**, no una predicción ex ante. No corresponde llamar fuga a usar resultados pasados en un clustering descriptivo.

Si los segmentos se usan después como variables de un predictor, deben calcularse únicamente con la historia disponible antes de cada predicción; de lo contrario, notas y entregas futuras sí filtrarían información.

## 7. Pruebas con la aplicación actual

El cargador real `backend/app/db/olist.py` leyó los seis archivos con los mismos conteos del inventario. Configuración resuelta: `D:\ProyectoFinalSWI\data\olist`.

| Prueba | Resultado |
|---|---|
| `GET /api/health` | HTTP 200; seis indicadores `datos_olist=true` |
| `GET /api/demand/forecast?horizonte=7&historico=14` | HTTP 200; 14 puntos históricos y 7 de pronóstico |
| Histórico recibido en esa consulta | 2018-08-02 a 2018-08-15, con conteos reales de Olist |
| Estado del pronóstico | `modelo_entrenado=false`; repetición semanal provisional, **no SARIMA entrenado** |
| `GET /api/reviews/seller/{id}?n_oraciones=3`, vendedor con mayor número de ítems | HTTP 200; 1.003 oraciones elegibles y resumen no vacío de 3 |

`health` detecta existencia de archivos; las pruebas de lectura y contenido son las que acreditan su carga. Retrasos, demanda y vendedores siguen sin modelos entrenados. El modelo de imágenes existente no utiliza estos CSV.

No se ejecutaron entrenamientos ni pruebas de precisión de modelos. No se modificaron los contratos HTTP.

## 8. Acciones priorizadas antes de entrenar

| Prioridad | Acción propuesta | Motivo |
|---|---|---|
| P0 | Aprobar si retraso significa superar el día prometido o su medianoche | Afecta a 1.292 etiquetas |
| P0 | Definir medias por pedido o por artículo en vendedores | Diferencias de hasta 38 puntos porcentuales en retrasos |
| P0 | Sustituir evaluación aleatoria por evaluación temporal con etiquetas maduras, según objetivo operativo | Evitar usar resultados futuros al evaluar |
| P1 | Tratar peso ausente/cero y pedidos multivendedor explícitamente | 16 faltantes agregados a cero y 509 pedidos con varios estados |
| P1 | Definir política de anomalías cronológicas, sin borrado automático | 1.373 pedidos elegibles afectados por alguna comprobación |
| P1 | Documentar ventana y significado de demanda; estudiar días sin cobertura | Cuatro ceros iniciales y cola final escasa |
| P1 | Preservar la cardinalidad de reseñas y documentar su agregación | `review_id` no es clave única |
| P2 | Evaluar varios cortes y baselines; justificar k y nombres de segmentos | No hay evidencia de calidad nueva sin entrenar/evaluar |
| P2 | Tratar textos vacíos y limitar el tamaño de resumen | TextRank es viable, pero no todos los mensajes son resumibles |
| P2 | Revisar compatibilidad futura de pandas | `.dt.dayofweek` emitió advertencia de deprecación; hoy no impidió el análisis |

Estas son recomendaciones: **no se aplicó ninguna corrección al código o a los datos**.

## 9. Trazabilidad de los archivos

SHA-256 calculados sobre los bytes originales, verificados nuevamente al finalizar el análisis:

| Archivo | SHA-256 |
|---|---|
| `olist_orders_dataset.csv` | `8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf` |
| `olist_customers_dataset.csv` | `983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2` |
| `olist_order_items_dataset.csv` | `0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279` |
| `olist_products_dataset.csv` | `3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc` |
| `olist_sellers_dataset.csv` | `1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e` |
| `olist_order_reviews_dataset.csv` | `012b61c7593e34f51fa614efdf802b9c7056ce6aae5307ddb93236e7cfc797d7` |

Para reproducir los criterios principales, con el entorno de `backend/.venv`:

```python
# Desde la raíz del repositorio; comprobaciones en memoria, sin entrenamiento.
from pathlib import Path
import pandas as pd

data = Path("data/olist")
orders = pd.read_csv(data / "olist_orders_dataset.csv")
for col in [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]:
    orders[col] = pd.to_datetime(orders[col], errors="raise")

eligible = orders[orders.order_status.eq("delivered")].dropna(
    subset=["order_delivered_customer_date", "order_estimated_delivery_date"]
)
late_timestamp = (
    eligible.order_delivered_customer_date > eligible.order_estimated_delivery_date
)
late_day = (
    eligible.order_delivered_customer_date.dt.normalize()
    > eligible.order_estimated_delivery_date.dt.normalize()
)
print(len(eligible), late_timestamp.sum(), late_day.sum())  # 96470, 7826, 6534

daily = (
    orders.order_purchase_timestamp.dt.floor("D").value_counts()
    .sort_index().asfreq("D", fill_value=0)
    .loc["2017-01-01":"2018-08-15"]
)
print(len(daily), daily.sum(), daily.eq(0).sum())  # 592, 96912, 4
```

El script de auditoría y su perfil JSON detallado se ejecutaron en el directorio temporal de la sesión. El único archivo creado dentro del repositorio para este trabajo es este informe.
