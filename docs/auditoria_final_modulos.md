# Auditoría final de los cinco módulos de CommerceAI

**Fecha:** 24 de noviembre de 2026. **Base auditada:** `main`, `007c618efa3cc2b8a4c9be11310751f76d5d4d3d`.

## Activación en `feature/completar-modulos-ia`

| Módulo | Evidencia de activación | Limitación |
|---|---|---|
| 1 · ETA | Se conserva el HGB de regresión aprobado. Agosto retrospectivo: MAE 3,31 días, RMSE 4,33. | No es una prueba externa; subestima entregas largas. |
| 2 · Demanda | SARIMAX(2,1,2)(1,1,1)[7] con calendario. Holdout cronológico de 28 días: MAE 46,29 y RMSE 57,54 pedidos/día, frente a 49,64 y 63,34 al repetir la última semana. Cobertura nominal 80 %: 23/28 días (82,14 %). | Un solo holdout reutilizado, no una prueba nueva. El reporte histórico perdió en ese mismo holdout (MAE 60,99); el ajuste actual usa statsmodels 0.15.0 y no demuestra una mejora general frente al baseline. No hay demanda reciente: los 90 días de la UI son una proyección desde agosto de 2018; las bandas son diarias y no califican al total. |
| 3 · Imágenes | CNN reentrenada y servida: accuracy 92,14 %, precisión macro 92,14 %, recall macro 92,14 % y F1 macro 92,13 % en las 10.000 imágenes oficiales. El SHA-256 devuelto por FastAPI coincide con el manifiesto. | No hay detector de fuera de dominio ni garantía sobre fotografías comerciales. |
| 4 · Vendedores | K-Means agrupó 1.766 vendedores: 627 de alto volumen, 837 estables y 302 con mayor retraso. El mejor Silhouette entre k=2–8 fue k=3 (0,3053; k=4: 0,2515). | Separación moderada, sin etiquetas objetivo; el gráfico y la tabla muestran los 300 vendedores de mayor facturación. |
| 5 · Reseñas | Se conserva TextRank + MMR sobre texto manual u Olist; sin archivo de modelo. | Es extractivo; no verifica hechos ni corrige sesgo de la muestra. |

Demanda se ancla en `2018-08-18`, tanto en el modo entrenado como en el fallback sobre Olist. El rango útil
es `2017-01-05`–`2018-08-18` (591 días, 97.687 pedidos). Métricas y huella del bundle en
[`demanda_metricas.json`](demanda_metricas.json). La evaluación histórica de desarrollo queda en
[`resultados_demanda/metricas.json`](resultados_demanda/metricas.json): MAE 42,68 / RMSE 51,42 frente a
48,67 / 59,95 estacional y 40,98 / 50,78 del promedio por día de semana.

Los segmentos usan volumen, facturación, nota, retraso y días de entrega medios. El perfil de alto volumen
registra 127,27 pedidos de media; el estable tiene la menor tasa media de retraso (≈3 %) y la mayor nota
(4,38/5); logística problemática tiene ≈18 % de entregas tardías y 17,37 días medios. Son descripciones
relativas, no reglas de causalidad ni sanción. Véanse los
[perfiles, preparación y huellas del par exportado](activacion_vendedores_metricas.json).
Una segunda ejecución local reprodujo el SHA-256 de los dos archivos.

El binario de la publicación original no apareció en el árbol ni en releases o artefactos de GitHub.
La copia local anterior se respaldó y se reevaluó en el test oficial: accuracy **91,52 %** y F1 macro
**0,91453** ([resultados y hashes anteriores](clasificacion_productos_previa_local.json)). Esas cifras
no corresponden al 92,46 % publicado para el otro binario. La nueva CNN se entrenó con semilla 42,
54.000 imágenes de ajuste y 6.000 de validación estratificada; se eligió la época 20 por su accuracy
de validación (92,83 %). Se verificaron los archivos oficiales descargados y se reprodujo el resultado
de test cargando el TorchScript exportado. [Manifiesto activo](clasificacion_productos_metricas.json):
precisión, recall y F1 por clase, además de la matriz de confusión.

`POST /api/products/classify` devuelve `X-Model-SHA256` con
`8d21032cd80b01be3deb7bffc342eef87b7eeff2f3dd3b526959481e86be52c2`. El mismo archivo produjo
**Botín (99,91 %)** y, con el *fixture* original de camisa ambigua, **Camisa (56,20 %)** con aviso
de revisión. No se cambió el *fixture* ni el umbral del 70 %. Las seis fotos de ejemplo se clasificaron
en la categoría esperada; son casos de integración, no una evaluación representativa de catálogo.

### Verificación final

- Python: **178 pruebas y 53 subpruebas aprobadas**, sin omisiones.
- FastAPI por HTTP: ETA con y sin fecha; SARIMAX a 7, 30 y 90 días; resumen y asignación de K-Means;
  CNN con confianza alta y baja y hash coincidente; TextRank manual y con datos de Olist.
- React: `npm run build` pasó. Playwright: **108 pruebas aprobadas**, sin omisiones, en cuatro tamaños.
  La prueba de accesibilidad de vendedores cubre el gráfico con datos reales.
- Los seis SHA-256 de Olist y el SHA-256 del modelo ETA coinciden con los originales. No se modificaron
  TextRank ni la carpeta duplicada.
- Avisos no bloqueantes: deprecación de TorchScript, tamaño del paquete Vite y reinicios de conexiones
  locales de Windows durante el E2E; las peticiones y pruebas verificadas pasaron.

### Preparar otra instalación

Desde la raíz, con Olist preparado según [`data/README.md`](../data/README.md) y el entorno Python activo:

```sh
python -m ml.train_demand --output candidatos/demand_model.joblib --metrics candidatos/demanda.json
python -m ml.train_sellers --models-dir candidatos/vendedores --metrics candidatos/vendedores.json
python -m ml.train_products --threads 4 --output candidatos/product_cnn.pt --metrics candidatos/productos.json
```

Usar rutas nuevas. Comparar cada salida con su propio informe; una reejecución en otro entorno no garantiza
hashes idénticos. Crear un respaldo antes de instalar en `backend/app/models/` los candidatos verificados y
reiniciar la API; el CSV y joblib de vendedores se instalan juntos. Para ETA conservar el artefacto aprobado
de [la integración](integracion_tiempo_entrega.md); no sustituirlo por el clasificador antiguo.

## Línea base histórica (`main`, antes de la activación)

**Dictamen inicial:** se demostraron de extremo a extremo **tiempo de entrega (1)** y **resumen de texto (5)**. **Imágenes (3)** ejecutaba una CNN real, pero la versión local no correspondía a las métricas publicadas. **Demanda (2)** y **clustering (4)** no tenían artefactos instalados. Las secciones siguientes conservan esa fotografía inicial, no el estado de la rama.

## 1. Repositorio, alcance y fuentes

`HEAD`, `origin/main` y `refs/heads/main` de GitHub coincidieron. El árbol estaba limpio al inicio. GitHub registra ambos PR como `MERGED` y sus commits están en `main`.

| Integración | Aporte inspeccionado |
|---|---|
| [PR #3 — demanda](https://github.com/antorm08/ProyectoFinalSWI/pull/3), `ec6efa4` | 31 archivos del PR: ventana de cobertura y serie diaria, selección temporal y calibración, especificación SARIMAX y bundle, HTTP y tipos, React y gráfico/tabla, pruebas, reportes, capturas y manuales. Véanse [`ml/demand_series.py`](../ml/demand_series.py), [`ml/evaluate_demand.py`](../ml/evaluate_demand.py), [`ml/train_demand.py`](../ml/train_demand.py), [`demand_model.py`](../backend/app/services/demand_model.py), [`demand_calendar.py`](../backend/app/services/demand_calendar.py), [`demand_service.py`](../backend/app/services/demand_service.py) y [evaluación](evaluacion_demanda.md). |
| [PR #4 — reseñas](https://github.com/antorm08/ProyectoFinalSWI/pull/4), `007c618` | 17 archivos del PR: TextRank ponderado + MMR, temas por léxico, resumen ejecutivo por plantilla, consultas por vendedor/categoría/plataforma, tipos HTTP, React, evaluación comparativa, pruebas, capturas y manuales. Véanse [`summary_service.py`](../backend/app/services/summary_service.py), [`reviews.py`](../backend/app/api/routes/reviews.py), [`Reviews.tsx`](../frontend/src/pages/Reviews.tsx) y [evaluación](evaluacion_resumen.md). |

Se distinguió **evidencia ejecutada** (suite, HTTP/inferencia y conteos recalculados), **evidencia publicada** (métricas en JSON, informes y capturas de entrenamientos previos) y **código inspeccionado** (funcionalidad que no se pudo activar). Un HTTP 200 o `/api/health` en verde no acredita que se haya ejecutado un modelo entrenado. No se ejecutó ningún script de entrenamiento o evaluación que volviera a ajustar modelos de entrega. Las pruebas existentes sí usan sus propios ajustes mínimos sobre *fixtures* en memoria; no generan artefactos de producto. No se modificaron código, CSV, modelos ni la carpeta duplicada. Este informe es el único archivo añadido al árbol principal.

### Datos y artefactos

- **Olist:** seis CSV presentes; los seis SHA-256 coinciden con [la auditoría de datos, §9](validacion_olist.md#9-trazabilidad-de-los-archivos). Se recalcularon 99.441 pedidos, 96.470 entregas elegibles para ETA, 1.766 vendedores elegibles y 99.224 reseñas (40.950 con comentario). No son datos sintéticos.
- **Fashion-MNIST:** fuente y protocolo documentados en [la evaluación CNN](evaluacion_productos.md#1-datos-y-protocolo) y [`data/README.md`](../data/README.md). `data/fashion_mnist/` contiene solo `.gitkeep`; los *fixtures* de ropa y ejemplos de React existen, pero no sustituyen el conjunto oficial. No se descargó ni se reevaluó en la auditoría.
- Debajo de `backend/app/models/` Git rastrea solo `.gitkeep`; `.gitignore` excluye el resto. Inventario local:

| Archivo | Estado y uso efectivo |
|---|---|
| `delivery_regression_experimental.joblib` | Presente, 326.117 B; SHA-256 `4a698a3771aa50c0e945d24db434fa7b074b3a16823201128966ba54d804df20`. Es el aprobado y servido por ETA. |
| `delay_model_exploratorio.joblib` | Presente, 449.874 B; SHA-256 `793b4a82e037d2541f5e02faafa0b7932a2e3f48fba2763562cd475ef096f315`. Clasificador HGB de investigación, no activo. |
| `delay_model_mejora_experimental.joblib` | Presente, 2.699 B; SHA-256 `6e086a77a83d36a2b50368eb5502c0d37aa3612d198bea87e18b00234dcbf2d8`. Logística de investigación, no activa. |
| `delay_model.joblib` | Ausente. Solo la ruta histórica usa una heurística explícita. |
| `demand_model.joblib` | Ausente al auditar `main`. Su reporte original en `007c618` documenta uno de 28.630 B, esquema 2, SHA-256 `433195681cfa1be618be7b036f183bb5313d5c87063ca18da3977215a7c8825f`, que no estaba disponible para verificar. |
| `product_cnn.pt` | Presente, 1.717.973 B. Inferencia real, pero el hash local difiere del publicado (véase módulo 3). |
| `seller_kmeans.joblib`, `seller_clusters.csv` | Ambos ausentes. El primero es necesario para asignar; el segundo para cargar segmentos y estadísticas en React. |
| TextRank | No requiere pesos, entrenamiento ni archivo de modelo. Depende del texto recibido o de Olist según el modo. |

## 2. Matriz de cumplimiento académico

**Cumple** = evidencia suficiente; **Parcial** = existe, pero falta una parte relevante o la validación no corresponde a la instalación; **No cumple** = la capacidad requerida no está operativa; **No verificado** = no se pudo comprobar la condición. La última fila incluye preparar el módulo en otro equipo, no solo que exista un archivo Markdown.

| Criterio | 1 · ETA | 2 · Demanda | 3 · Imágenes | 4 · Clustering | 5 · Texto |
|---|---|---|---|---|---|
| 1. Problema real definido | Cumple | Cumple | Cumple | Cumple | Cumple |
| 2. Técnica de IA identificada | Cumple | Cumple | Cumple | Cumple | Cumple |
| 3. Dataset real | Cumple: Olist | Cumple: Olist | Parcial: Fashion-MNIST documentado, test local ausente | Cumple: Olist | Cumple: Olist y entrada manual |
| 4. Preparación de datos | Cumple | Cumple para ajuste; parcial en fallback | Cumple | Cumple | Cumple |
| 5. Algoritmo implementado | Cumple | Cumple | Cumple | Cumple | Cumple |
| 6. Modelo entrenado, cuando aplica | Cumple: binario verificado | Parcial: ajuste documentado, binario no disponible | Parcial: binario funcional, procedencia no verificada | Parcial: ajuste/evaluación histórica, binario no disponible | Cumple: no requiere entrenamiento |
| 7. Evaluación con métricas apropiadas | Cumple para experimento, sin prueba externa nueva | Cumple, con reservas sobre desempeño y cobertura | Parcial: métricas del modelo publicado, no del local | Parcial: métricas internas; la elección de cuatro grupos no está justificada del todo | Cumple: comparación y evaluación descriptiva, sin anotación humana |
| 8. Integración con FastAPI | Cumple | Parcial: fallback ejecutado; contrato entrenado solo con fixtures | Parcial: inferencia, sin comprobar identidad del modelo | Parcial: rutas activas pero sin resultados de clustering | Cumple |
| 9. Funcionamiento desde React | Cumple: días y fecha opcional | Parcial: histórico real + fallback; banda del artefacto publicado no verificada hoy | Parcial: top 3 funciona, caso de baja confianza no coincide | No cumple en datos reales hoy: solo estado vacío | Cumple |
| 10. Documentación y reproducibilidad | Parcial: guía específica, falta integrar en los manuales | Parcial: guía, faltan binario de exposición y corregir lectura de banda | Parcial: guía detallada, sin binario de referencia local | Parcial: informe de evaluación, falta guía reproducible de demo | Cumple en el módulo; ver carencias del paquete general |

## 3. Evidencia por módulo

### 1 · Predicción de tiempo de entrega — funciona con modelo entrenado

**Problema:** estimar duración desde la compra hasta la entrega, no probabilidad de incumplimiento. La vista `/retrasos` llama `POST /api/delay/predict-duration`. El servicio valida el hash antes de deserializar y comprueba `Pipeline → ColumnTransformer → HistGradientBoostingRegressor`. El pipeline usa precio, flete, peso, cantidad, mes, día de semana, mismo estado, origen representativo y destino. Los pesos cero se convierten en NaN y las categorías se codifican. El plazo prometido figura en metadatos del DataFrame de ajuste, pero **no entra en el transformador ni se exige en el endpoint**.

- **HTTP real:** pedido de BRL 120 + 20 de flete, 1.500 g, un artículo, SP → RJ, compra `2018-08-06`: **10,908879 días**, llegada `2018-08-16`. Con agosto/lunes y sin fecha, igual duración y llegada `null`. Enviar `dias_estimados` a ETA da **422**; con artefacto simulado como ausente da **503** y ninguna predicción. Playwright comprueba el flujo de React en los cuatro tamaños y que editar retira la respuesta anterior.
- **Clasificación histórica separada:** `POST /api/delay/predict` conserva contrato y usa reglas cuando falta `delay_model.joblib`; el ejemplo devolvió 0,175 con `modelo_entrenado=false`. No la llama el formulario principal y no debe mostrarse como salida entrenada. Los experimentos históricos HGB (ROC-AUC 0,6632; AP 0,1087 en agosto) y logística de retraso son otra tarea, no comparables directamente con el MAE de ETA.
- **Evaluación publicada, ligada al binario verificado:** MAE macro de desarrollo **4,60 días**, RMSE **6,33**; agosto retrospectivo **3,31 días** de MAE, **4,33** de RMSE, R² **0,005** y **77,72 %** dentro de cinco días. En el mismo agosto, mediana histórica **4,59 días** y promesa Olist exacta **8,14 días** de MAE. Es menor error **puntual**, no una promesa mejor ni una fecha garantizada. En las 11 entregas de más de 30 días subestimó **25,45 días** en promedio. Agosto ya era conocido, se excluyó para seleccionar; no acredita generalización externa. El MAE mostrado es de cohortes, no intervalo por pedido.
- **Preprocesamiento auditado:** `dias_entrega` fraccional desde `compra` a entrega; una fila por pedido entregado con fechas válidas; joins con cardinalidad; peso faltante si algún artículo no tiene peso y origen representativo por valor. Cortes por compra y etiquetas disponibles antes de evaluar; persisten censura y sesgo hacia pedidos finalmente entregados.

Fuentes: [`common.py`](../ml/common.py), [`train_delay.py`](../ml/train_delay.py), [`compare_delivery_regression.py`](../ml/compare_delivery_regression.py), [`delivery_duration_service.py`](../backend/app/services/delivery_duration_service.py), [`Delay.tsx`](../frontend/src/pages/Delay.tsx), [evaluación ETA](evaluacion_tiempo_entrega.md), [métricas](resultados_tiempo_entrega/metricas.json) e [integración con evidencia](integracion_tiempo_entrega.md).

### 2 · Series temporales — funciona con fallback; SARIMAX no demostrable hoy

**Problema:** predecir pedidos creados por día calendario independientemente de su estado. `--check-data` reprodujo **591 días y 97.687 pedidos** del **2017-01-05 al 2018-08-18**, de una serie bruta de 774 días con 140 sin registros. El ajuste publicado usa **SARIMAX(2,1,2)(1,1,1)[7]** con festivos, Black Friday y días posteriores, en escala bruta. El bundle reconstruye con `smooth(params)` sin reoptimizar, conserva histórico y especificación y aporta banda empírica por semanas del horizonte. Localmente falta el bundle; código y pruebas de contrato cubren esa ruta con parámetros/artefactos de prueba, no prueban el binario publicado en esta máquina.

| Evidencia publicada | MAE (pedidos/día) | RMSE | MAE relativo a repetir semana |
|---|---:|---:|---:|
| SARIMAX, ocho orígenes de desarrollo | 42,68 | 51,42 | 0,866 |
| Repetir última semana, desarrollo | 48,67 | 59,95 | 1,000 |
| Media por día de semana, desarrollo | **40,98** | **50,78** | **0,842** |
| SARIMAX, últimos 28 días reservados | 60,99 | 71,79 | **1,229** |
| Repetir última semana, tramo reservado | **49,64** | — | 1,000 |

Los orígenes tienen ventana de ajuste expansiva, horizonte de 28 días y paso de 14; las 224 predicciones cuentan fechas repetidas entre orígenes, no 224 días independientes. Los seis orígenes de calibración son anteriores al tramo reservado y pronostican hasta 90 días. En el tramo **2018-07-22 a 2018-08-18**, la banda nominal de 80 % cubrió **75 %** y la teórica **46,4 %**. El modelo pierde frente al naive en ese tramo; la media por día de semana gana incluso en desarrollo. El protocolo se revisó y el holdout se recalculó; no es un test virgen. Queda autocorrelación semanal y el calendario declarado cubre 2016–2020. No se justifica uso prospectivo actual.

**Lo que hace hoy la API:** `GET /api/demand/forecast?horizonte=7|14|90&historico=7` devuelve 200 y el número correcto de puntos; la API acepta **1–90**, React ofrece **7–90**, y 91 devuelve **422** (antes máximo 180). El ancla local en fallback es `2018-10-17`: con siete días históricos se vio exactamente `[0, 0, 0, 0, 0, 1, 1]` y repite esos valores como pronóstico (**`modelo_entrenado=false`, sin banda**). El dato histórico es real, pero el fallback usa la **cola sin cobertura suficiente** que el entrenamiento excluye; presentarlo como demanda comercial fiable sería incorrecto. Si no hay Olist ni bundle, se activa otra ruta **sintética**; se comprobó aislando la disponibilidad en memoria, no es el modo de esta instalación. El gráfico y la tabla de `/demanda` se cargaron con histórico real y pronóstico provisional. La [captura registrada de banda](evidencia_demanda/banda-y-tabla-escritorio.png) es evidencia histórica separada, no de la ejecución auditada.

**Hallazgos de exposición:**

1. Se omiten **ocho** casos Playwright: dos (banda y métricas, y cambio de nivel) por cada uno de los cuatro tamaños. Su condición es `!health.modelos.demanda`. No son pruebas aprobadas.
2. [`DemandChart.tsx:9–19`](../frontend/src/components/DemandChart.tsx#L9) etiqueta la cobertura histórica `cobertura_80` con el nivel **seleccionado**. Si se elige 95 %, mostraría «banda del 95 %» usando la cifra de 80 %. Además, `published_metrics` combina MAE de desarrollo y cobertura del holdout: no son el mismo período.
3. [`Demand.tsx:21–24,46–48`](../frontend/src/pages/Demand.tsx#L21) suma bandas **diarias** y presenta un rango del total; no se ha estimado la cobertura conjunta del acumulado. MAE con prefijo «±» tampoco es intervalo. No defender esas leyendas como probabilidades de la totalidad del horizonte.
4. En la captura de navegador de la auditoría, el fallback mostró **8 pedidos en 30 días** y, aun sin `inferior/superior`, la leyenda «Banda 80 %». Las áreas se montan siempre en [`DemandChart.tsx:70–71`](../frontend/src/components/DemandChart.tsx#L70), por lo que la leyenda no prueba que exista una banda.

Fuentes: [evaluación](evaluacion_demanda.md), [JSON de comparación](resultados_demanda/metricas.json), [bundle publicado y métricas](demanda_metricas.json), [ruta](../backend/app/api/routes/demand.py), [gráfico](../frontend/src/components/DemandChart.tsx), [E2E](../frontend/tests/commerceai.spec.ts).

### 3 · Clasificación visual — CNN entrenada real, artefacto distinto

`POST /api/products/classify` recibe multipart `imagen`; `/productos` consume ese resultado real (categoría, top 3, confianza y aviso). El código de entrenamiento define una CNN Fashion-MNIST de **421.834 parámetros**, dos bloques convolucionales con normalización y *pooling*, *dropout* y dos capas densas, Adam, aumentos y selección por validación. Datos oficiales: 54.000 ajuste, 6.000 validación estratificada y 10.000 test. El servicio carga TorchScript en CPU; aplica EXIF, transparencia, gris, inversión/recorte respecto al fondo, escala 28×28 y normaliza con `0,2860/0,3530` antes de softmax. No devuelve resultados entrenados si falta el binario; tampoco dispone de un detector de objetos fuera del dominio.

| Comprobación | Resultado |
|---|---|
| Binario efectivamente cargado (`product_cnn.pt`) | SHA-256 `cd7877d089fd0986fe07b75e2ce6237043ce95c36be6fa089aa19b562ad9040d` |
| Binario de las métricas publicadas | SHA-256 `44aca28867b50caa2637b19a87aa0b92476f0e9ef1cea619b459026961e52129` |
| Métricas del publicado, no del local | Accuracy test **92,46 %**, F1 macro **0,92449**, validación **92,77 %**, F1 Camisa **0,7733**. Las pruebas de robustez con recortes y fondos simulados son intra-dominio, no fotos arbitrarias. |
| Inferencia HTTP del local | Botín **99,91 %**; prenda ambigua **Camisa 84,43 %**, `baja_confianza=false` con umbral 0,70. Imagen uniforme **422**, bytes inválidos **400**. |
| Fuera de dominio, comprobación diagnóstica | Figuras sintéticas sobre fondo blanco: cuadrado → **Bolso 99,95 %**; aspa → **Sandalia 71,22 %**; triángulo → **Vestido 87,49 %**. Las tres se aceptan sin aviso. No son imágenes de Fashion-MNIST ni una medición de tasa OOD. |

Los dos fallos Python son `test_artefacto_coincide_con_metricas_publicadas` (hash) y `test_fixtures_de_las_pruebas_e2e` (espera baja confianza). El mismo aviso falla en cuatro tamaños de Playwright. **La discrepancia de bytes es demostrable y anterior a PR #3/#4**: mismos dos fallos en el `main` previo `35298a2` con ese binario, y los PR no cambian el módulo, fixture ni sus aserciones de confianza. El binario carga en PyTorch `2.15.0+cpu`, también declarado en el informe publicado; no hay evidencia de incompatibilidad de versión como causa principal. No se identificó cuándo ni con qué protocolo se entrenó el archivo local, ni se pudo comparar su accuracy sin el dataset y el binario publicado. La cifra fija «92,5 %» en [`Products.tsx:96`](../frontend/src/pages/Products.tsx#L96) no debe atribuirse al binario local. Confianza softmax no es probabilidad calibrada de acierto.

Fuentes: [`train_products.py`](../ml/train_products.py), [`evaluate_products.py`](../ml/evaluate_products.py), [`product_service.py`](../backend/app/services/product_service.py), [informe](evaluacion_productos.md), [métricas y hash](clasificacion_productos_metricas.json), [`test_products.py`](../ml/tests/test_products.py).

### 4 · Clustering — evaluación histórica, sin segmentos servidos

**Problema:** describir vendedores por volumen, facturación de ítems, satisfacción y logística. La preparación sobre CSV reales reprodujo **110.189 ítems elegibles → 97.811 pares vendedor–pedido** y se mantuvieron **97.811** tras la unión de reseñas. Promedia reseñas por pedido antes de unir; nota, duración y tardanza se promedian por par, no por número de ítems. Hay 660 pares sin nota; de 2.970 vendedores con pedidos elegibles, 1.204 no llegan al mínimo de cinco y ninguno adicional quedó sin métricas tras ese mínimo: **1.766 vendedores**.

Implementado: `log1p` para pedidos y facturación, `StandardScaler` para las cinco variables y K-Means `n_init=20`, semilla 42. El entrenador imprime *silhouette* para k=2–7 y **fija k=4**; la evaluación recorre hasta k=8, y el mejor *silhouette* y otros índices internos son **k=3** (silhouette **0,3053**, CH **778,39**, DB **1,1225**). Para k=4: **0,2515**, CH **678,91**, DB **1,3713**; K-Means supera Ward (0,1672) y GMM (0,2001) en *silhouette* al comparar k=4, aunque Ward tiene mejor DB (1,298). ARI medio **0,9491** entre el ajuste base y cinco semillas sobre los mismos datos; no demuestra estabilidad ante nuevos vendedores, otras temporadas ni ponderaciones. Los JSON registran un ajuste efectuado en otra ejecución; **no hay modelo local cargable hoy**.

Perfiles publicados: **504** «Vendedores estrella (alto volumen)», **584** «Vendedores estables», **151** «Logística problemática» y **527** «Baja satisfacción». Las etiquetas son reglas de ranking sobre medias en escala original, no clases supervisadas: se nombra primero la mayor tasa de retraso, luego mayor facturación de los restantes, luego la menor nota y, finalmente, «estables». La etiqueta «Baja satisfacción» es relativa (media 4,0/5); no atribuye responsabilidad causal. En pedidos multivendedor una reseña de pedido puede entrar en cada vendedor participante; facturación de artículos no equivale a liquidación.

**HTTP y vista:** `GET /api/sellers/clusters` → **200**, `modelo_entrenado=false`, `clusters=[]`, `vendedores=[]`; `POST /api/sellers/assign` → **200**, `cluster=-1` y «modelo no entrenado». **No hay datos simulados.** `/vendedores` abre, muestra estado «Segmentación no disponible» y el código define las tarjetas, las medias, tabla y gráfico de satisfacción vs. tardanza por segmento. No se verificó el gráfico con datos entrenados porque falta `seller_clusters.csv`; al tenerlo la gráfica muestra el subconjunto de **hasta 300** por facturación, no toda la población. El resumen depende solo de la tabla pero `health.modelos.vendedores` solo del bundle: pueden discrepar. Entregar los dos archivos como par con misma procedencia y probar ambos endpoints. La UI no llama a `/assign`; ello no impide el requisito de mostrar clustering en la interfaz.

Fuentes: [`train_sellers.py`](../ml/train_sellers.py), [`evaluate_sellers.py`](../ml/evaluate_sellers.py), [evaluación](evaluacion_sellers.md), [JSON](evaluacion_sellers_metricas.json), [`seller_service.py`](../backend/app/services/seller_service.py), [`Sellers.tsx`](../frontend/src/pages/Sellers.tsx).

### 5 · Resumen extenso — funciona con datos reales y texto manual

**Técnica comprobada:** limpia/normaliza y divide en oraciones de al menos tres palabras; agrupa por clave sin acentos, mayúsculas o puntuación y cuenta en cuántas reseñas aparece. TF-IDF de caracteres de 3–5, preselección de hasta 400 candidatas, grafo de similitud y PageRank ponderado por frecuencia; MMR (λ=0,7) penaliza similitud y evita paráfrasis mientras hay alternativas. En texto pegado, conserva el orden de las frases seleccionadas. El ejecutivo español usa **plantillas de indicadores**, no LLM. Reclamos/elogios se separan por **nota 1–2 / 4–5**, no por un clasificador de sentimientos; diez temas definidos por léxicos en español/portugués. No requiere entrenamiento ni artefactos.

- **Texto manual extenso:** petición de prueba de **7.040 caracteres / 90 oraciones elegibles**, HTTP **200**, devolvió **cinco** citas en orden original. Aparecieron anuncio, embalaje/puntualidad y atención/retraso. Tres citas eran variantes del mismo patrón de anuncio, caso límite de redundancia semántica pese a deduplicación exacta. No se presentó esta entrada sintética como reseñas de Olist.
- **Olist real:** `GET /api/reviews/overview` dio **99.224 / 40.950** (total / con comentario). `seller/2eb70248d66e0e3ef83659f71b244378`: **199 / 125**. `category/cama_mesa_banho`: **9.432 / 4.320**. `/options`: **30 vendedores y 73 categorías**. En plataforma se observaron «Não recebi o produto.» (**132**) y «Chegou antes do prazo!» (**166**); junto al primero apareció una variante de la misma queja (**73**). El ejecutivo y las frases son representativos, no eliminan toda repetición ni garantizan relevancia humana.
- **Duplicación:** se deduplica por `review_id,order_id`; la selección por vendedor/categoría usa pedidos únicos y `isin`, no expande reseñas por artículo. Una reseña puede pertenecer a varios ámbitos distintos, así que sus poblaciones no son sumables. La nota neutra cuenta en totales, no en las secciones de citas. Lexicón y proxies fácticos no equivalen a etiquetas de verdad de sentimiento o temas.
- **Calidad publicada:** 120 grupos, TextRank ponderado + MMR frente al clásico: cobertura **0,790 vs. 0,774**, temas **73,6 % vs. 60,4 %**, redundancia **0,305 vs. 0,429**, reseñas representadas **20,0 % vs. 18,8 %**. Tema «pedido no recibido»: **67,8 % vs. 26,1 %**, lift **2,59**, comprobado con timestamps en la evaluación anterior. Son métricas proxy con grupos seleccionados, no una evaluación humana independiente o causal.
- **Errores y React:** vendedor y categoría inexistentes y ausencia de dataset probada en memoria devuelven **404** en análisis por datos. `/resenas` ofrece cuatro modos, carga, errores, aviso de falta de categorías y estado sin oraciones, frases en portugués y copiado. El texto manual funciona sin Olist. En caso de error de `/options`, la categoría puede quedar deshabilitada y el mensaje sugiere datos faltantes sin distinguir el error de red; la disponibilidad de `products` también afecta categorías y no se comprueba en `data_available()`. Es un límite del modo sin datos, no un fallo en el entorno auditado.

Fuentes: [`summary_service.py`](../backend/app/services/summary_service.py), [`evaluate_reviews.py`](../ml/evaluate_reviews.py), [evaluación](evaluacion_resumen.md), [métricas](evaluacion_resumen_metricas.json), [API](../backend/app/api/routes/reviews.py), [React](../frontend/src/pages/Reviews.tsx), [pruebas](../ml/tests/test_reviews.py).

## 4. Resultados de QA, causas y alcance

Entorno de ejecución: Python **3.12.10**, pandas **3.1.1**, scikit-learn **1.9.1**, statsmodels **0.15.0**, Torch **2.15.0+cpu**. Dependencias de prueba ya preparadas en la copia temporal sin instalar globalmente. `pytest` se ejecutó sobre el repositorio principal sin caché de pytest ni bytecode. Build y navegador usaron la copia temporal ya existente con **el mismo árbol Git `6393ad7…`**, sus dependencias locales y copias de los seis CSV y los artefactos disponibles. Tamaños de Playwright: 1440, 1024, 768 y 375 px.

| Verificación | Resultado y alcance |
|---|---|
| `python -B -m pytest ml/tests -q -p no:cacheprovider` | **171 aprobadas, 2 fallidas, 1 omitida**; 53 subpruebas aprobadas. Los dos fallos son exclusivamente identidad y aviso del binario CNN; reproducidos antes de la fusión. La omitida requiere el test de Fashion-MNIST local. |
| `python -B -m ml.train_demand --check-data` y preparación de vendedores | **Aprobadas**, con conteos reales; sin ajuste ni escritura de artefactos. |
| Contratos e inferencia por FastAPI/ASGI | **Ejecutados**: ETA 200/422/503 y clasificación histórica, demanda 1/7/14/90/91 y modo sintético aislado, imágenes multipart y errores 400/422, ambos endpoints de clustering, texto manual y reseñas 200/404. No se inventaron respuestas de éxito con artefactos ausentes. |
| `npm.cmd run build` (TypeScript + Vite) | **Aprobada**. Bundle JS **756,73 kB** sin comprimir; aviso de chunk >500 kB, no fallo. |
| `npm.cmd run test:e2e` (104 casos, ~4 min) | **92 aprobadas, 4 fallidas, 8 omitidas**, mismo resultado que en la fusión de prueba. Falla el mismo caso de confianza CNN en cada tamaño; ocho casos entrenados de demanda se excluyen por falta del artefacto. Pasan ETA con modelo, demanda provisional y tabla, reseñas, las seis vistas y las auditorías de accesibilidad de la suite. |

No se usó el número de pruebas omitidas como aprobación. Los problemas de procedencia CNN ya existían en `35298a2`. Los fallos semánticos de la cobertura presentada a 95 % y del rango acumulado se identificaron por inspección en el código de UI de **PR #3**; las pruebas no garantizan el texto del nivel alternativo con métricas reales. La cola pobre de Olist y la falta de artefactos son riesgos de datos y distribución, no conflictos de Git. No se halló una regresión funcional nueva de PR #4 en los casos reales ejecutados.

## 5. Entrega académica y plan mínimo

El **código fuente** de los cinco algoritmos, API, vistas, evaluadores y pruebas está versionado. Los informes por módulo ya aportan evidencia. El **paquete documental general está incompleto**: [`README.md`](../README.md#L5) describe M1 como riesgo de retraso y recomienda `train_delay.py`, que no genera el ETA servido; su secuencia de `cd ../ml` y `python -m ml…` alterna directorios incompatibles si se ejecuta literalmente. [Manual de instalación](manual_instalacion.md) tiene secciones de backend/frontend/Docker vacías; faltan instrucciones de artefacto aprobado de ETA y del par de clustering. [Manual de usuario](manual_usuario.md) tiene encabezados sin contenido para panel, M1 y clustering. [Informe técnico](informe_tecnico.md) conserva plantilla, M1 de clasificación y §2.4.4 sin desarrollar; arquitectura, ventajas y conclusiones quedan incompletas. No se debe declarar completa la entrega solo porque existan los nombres de archivo.

| Prioridad | Tarea para una siguiente fase (no realizada en esta auditoría) | Archivos / evidencia de aceptación |
|---|---|---|
| **P0** | Conseguir o reproducir de forma autorizada el bundle SARIMAX de referencia; no mostrar el fallback de cola como modelo. Verificar nombre, esquema, hash/procedencia y valores de histórico/futuro y banda a 7, 14 y 90. | `backend/app/models/demand_model.joblib`, `docs/demanda_metricas.json`, `frontend/tests/commerceai.spec.ts`. `modelo_entrenado=true`, ancla coherente y ejecutar los ocho casos ahora omitidos. |
| **P0** | Verificar el CNN: recuperar **el binario exacto publicado** o reevaluar y publicar la versión que realmente se distribuya. No cambiar el hash esperado para ocultar la divergencia. | `product_cnn.pt`, Fashion-MNIST, `ml/tests/test_products.py`, Playwright, `docs/clasificacion_productos_metricas.json`, `Products.tsx`. Pruebas de identidad, test oficial y resultado esperado en ejemplos. |
| **P0** | Preparar y entregar juntos el bundle y la tabla de **clustering**; documentar si se mantiene k=4 por interpretabilidad o se adopta k=3 con nueva evaluación. Probar asignación, perfiles y dispersión reales. | `ml/train_sellers.py`, `ml/evaluate_sellers.py`, `seller_kmeans.joblib`, `seller_clusters.csv`, `docs/evaluacion_sellers*`, `Sellers.tsx`. Contar 1.766 filas, comprobar etiquetas y que el gráfico muestre su subconjunto sin duplicaciones. |
| **P0** | Completar manuales e informe (el enunciado los exige). Corregir denominación/arranque de ETA; congelar versión de Python y dependencias compatibles y registrar hashes/instalación de todos los artefactos de la exposición. | `README.md`, `docs/manual_instalacion.md`, `docs/manual_usuario.md`, `docs/informe_tecnico.md`, `backend/requirements.txt` / `ml/requirements.txt` si hace falta; reproducir en una instalación limpia antes de entregar. |
| **P1** | Corregir interpretación de bandas y fallback: separar métricas de desarrollo y holdout, etiquetar solo cobertura contrastada, quitar el rango agregado si no se calibra, cortar la cola también en modo Olist provisional o advertirlo. | `DemandChart.tsx`, `Demand.tsx`, `demand_service.py`, `schemas.py`, manual y pruebas de contrato/UI. Si no se corrige antes de la demo, mostrar la banda a 80 % y explicar explícitamente sus límites, sin presentar el agregado como intervalo validado. |
| **P1** | Preparar una navegación de exposición con ejemplos fijos, Olist y modelos verificados antes de abrir el navegador. Añadir pruebas E2E del gráfico de clustering entrenado; distinguir «archivo detectado» de «servido» y casos sin datos. | `backend/app/api/routes/health.py`, `seller_service.py`, `frontend/tests/commerceai.spec.ts`, guion de manual. Para reseñas: texto largo, plataforma y categoría; precalentar la primera consulta o prever espera. |
| **P2** | Mejoras metodológicas: intervalos ETA y nuevo período maduro; baseline por día de semana con bandas para demanda; abstención/calibración y fotos fuera del dominio para CNN; estabilidad del clustering fuera de semilla; revisión humana de resúmenes y manejo de errores de opciones. | Evaluadores y servicios respectivos. No son requisitos de producción ni deben bloquear una demostración académica honesta con los P0 cerrados. |

**Respuesta final.** Presentaría **1 (ETA) y 5 (TextRank)** como terminados y demostrables en esta instalación, con sus limitaciones explícitas. **3** se puede mostrar como inferencia experimental de la CNN **local**, pero no como demostración del **92,46 %** publicado hasta alinear el artefacto. **2** requiere habilitar y comprobar el SARIMAX entrenado; sin él solo se muestra un baseline provisional sobre una cola escasa. **4** requiere los dos artefactos y una prueba con segmentos reales en pantalla. Para concluir CommerceAI falta, como mínimo, **cerrar esas tres cadenas de artefactos**, repetir la demo integrada y **completar manuales e informe técnico**. No se entrenó, corrigió, sobrescribió ni publicó nada durante la auditoría.
