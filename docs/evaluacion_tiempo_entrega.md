# Evaluación experimental del tiempo de entrega

Este experimento del **Módulo 1** estima la duración de entrega de un pedido individual en Olist. No pronostica volumen diario (Módulo 2/SARIMA), no reemplaza la clasificación de retrasos y no modifica `POST /api/delay/predict`.

## Objetivo y preparación

```python
dias_entrega = (entrega_real - compra).total_seconds() / 86400
```

Se predicen **días transcurridos fraccionarios**, no días calendario truncados, ni probabilidad de retraso. Se reutilizan `delivered_orders()` y `build_dataset()` sin modificar sus implementaciones: una fila por pedido entregado con timestamps válidos y promesa calculable, uniones con cardinalidad comprobada, sumas de todos los artículos y origen representativo por mayor valor de artículos, luego cantidad y sigla. Se preserva como desconocido (NaN) el peso del pedido si algún artículo tiene peso faltante o cero. No se convierten cancelaciones en entregas, ni se borran extremos por duración o por anomalías auxiliares.

La implementación existente no admite una entrega anterior a compra. Una anomalía de orden de transportista/aprobación, por sí sola, no invalida el intervalo compra–entrega y solo se audita. Las fechas de entrega/aprobación/transporte, el estado final, las reseñas, los identificadores y el objetivo no se usan como entradas.

Se revisaron [la auditoría de Olist](validacion_olist.md), [la preparación](preparacion_olist.md), [la primera evaluación](evaluacion_retrasos.md), [la mejora de clasificación](mejora_retrasos.md), las utilidades de `ml` y los esquemas y la ruta FastAPI. `ml/temporal.py` documenta la política reutilizada; el script nuevo la reproduce con dos bloques en cada corte para no requerir un tercer bloque ficticio ni mezclar el conjunto final con selección.

### Distribución de la duración real

De 99.441 pedidos, 96.478 figuran como entregados y ocho de ellos no tienen fecha real. La muestra queda en **96.470 pedidos**, sin pérdidas en los joins; conserva 22 pesos desconocidos y 509 pedidos multiorigen. Las 625 cancelaciones quedan fuera, incluso si hay fecha real registrada.

| Mín. | P5 | P25 | Mediana | Media | P75 | P90 | P95 | P99 | Máx. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,53 | 3,02 | 6,77 | 10,22 | 12,56 | 15,72 | 23,10 | 29,27 | 46,05 | 209,63 |

Hay 13 entregas de menos de un día, 4.550 de más de 30, 306 de más de 60 y 77 de más de 90. La cola larga explica por qué MAE, RMSE y error de los casos prolongados deben leerse juntos. No hay entregas anteriores a la compra. En 1.373 pedidos hay alguna anomalía auxiliar de orden: transportista anterior a compra (165) o aprobación (1.350), entrega al cliente anterior a aprobación (61) o al transportista (23); los conteos se solapan. Faltan 14 aprobaciones y una fecha de transportista en la muestra elegible. No se excluyen por estos motivos.

La mediana es **6,55 días** dentro del mismo estado representativo (34.696 pedidos) y **12,79 días** entre estados (61.774). Por estado del cliente: SP 7,21 (40.494), MG 10,31 (11.354), RJ 12,04 (12.350), BA 16,91 (3.256) y AM 25,88 (145). Son diferencias descriptivas y las muestras pequeñas no soportan una comparación causal. El JSON incluye cada estado de cliente y de vendedor. En cohortes completas la mediana cae de 14,26 días en febrero a 9,31 en abril, 7,96 en junio y 7,01 en agosto: hay deriva relevante para el ETA.

## Predictores y modelos

Precio de artículos, flete, peso total, número de artículos, estados de cliente y vendedor representativo, mismo estado, mes, día de semana y días prometidos calendario. Se supone que el plazo de Olist ya está disponible al hacer la predicción. Para un caso que preceda a la publicación de esa promesa se ofrece una ablación de HGB sin ese campo, no una imputación artificial.

La variante HGB derivada evalúa precio, flete y peso por artículo, `flete/(precio+0,01 BRL)` y pertenencia a la misma macrorregión brasileña a partir de las dos siglas. No se infieren coordenadas, distancias de carretera ni rutas reales. Los datos tienen prefijos postales, pero esta copia no aporta una geocodificación histórica validada; tampoco se añadirán campos al contrato para este experimento. En pedidos multiorigen la geografía representa solo el origen elegido.

| Candidato | Configuración |
|---|---|
| DummyRegressor | Mediana de duración del bloque de ajuste: referencia de ausencia de información para MAE |
| Ridge | Regularización L2, `alpha=10` |
| Random Forest | 65 árboles, profundidad máxima 14, hoja mínima 16, 70 % de variables por división |
| HistGradientBoosting (HGB) | Pérdida absoluta, 220 iteraciones, 31 hojas, mínimo 40 pedidos/hoja, tasa 0,07, L2=2; sin parada temprana aleatoria |
| HGB + derivadas / HGB sin promesa | Misma configuración, cambiando solo las variables indicadas |
| XGBoost | Pérdida absoluta, 220 árboles hist, profundidad 5, tasa 0,07, muestras 90 %, columnas 85 %, L2=2 |
| Olist, timestamp exacto | Tiempo entre compra y fecha estimada registrada (medianoche), en días fraccionarios |
| Olist, días calendario | Diferencia entre fechas normalizadas, coherente con los días enteros del contrato actual |

Todas las alternativas aprendidas usan un pipeline de scikit-learn con one-hot de estados, categorías desconocidas ignoradas y selección explícita de columnas. Dummy, Ridge y Random Forest usan mediana aprendida solo en ajuste, con indicador de faltante cuando existe en ese bloque; solo Ridge escala. Los otros árboles usan NaN nativo. Las predicciones negativas de los modelos se recortan a cero para las métricas; no se recorta ni elimina el objetivo. Semilla 42, hasta dos hilos y configuraciones fijas, sin búsqueda exhaustiva. Olist no se reentrena ni corrige retrospectivamente.

La estimación de Olist es una fecha prometida, no necesariamente una predicción puntual de la media o la mediana logística. Puntuarla como número de días es una referencia útil, pero reducir ese error no demuestra que se deba hacer una promesa más corta ni que se mantenga el nivel de servicio prometido.

## Diseño temporal

Límites de compras `[inicio, fin)`. Las etiquetas de entrenamiento se exigen **estrictamente anteriores al inicio de evaluación**. Dejar un mes entre las últimas compras de entrenamiento y el inicio de evaluación reduce pero no elimina la censura. La evaluación de cada cohorte también exige que la entrega se haya observado antes de su cierre. No conocemos el timestamp de ingestión; se aproxima por entrega real.

| Corte | Compras de ajuste anteriores a | Compras de evaluación | Cierre de etiquetas de evaluación |
|---|---|---|---|
| D1 | 2018-01-01 | febrero 2018 | 2018-04-01 |
| D2 | 2018-03-01 | abril 2018 | 2018-06-01 |
| D3 | 2018-05-01 | junio 2018 | 2018-08-01 |
| Retrospectivo | 2018-07-01 | agosto 2018 | 2018-10-18 |

La regla fijada fue menor **MAE macro** de los tres cortes de desarrollo (peso igual por mes), luego RMSE macro. Modelos y referencias comparten exactamente los pedidos en cada corte. Agosto se separó antes de ajustar o elegir; tras congelar el ganador, se reajustaron las configuraciones predefinidas con los datos disponibles en el último corte y se evaluaron en agosto para comparar, sin cambiar la selección.

**Agosto ya se consultó para clasificación: no es una prueba externa inédita.** No se usa la cola de septiembre/octubre como sustituto: tiene muy pocas compras. Compradores y vendedores pueden reaparecer en distintos períodos. El estudio mide pedidos finalmente entregados, no tiempo hasta la cancelación ni casos todavía abiertos. Los recuentos de duraciones de compras inmaduras son retrospectivos; sus resultados no se pasan al modelo.

| Evaluación | Ajuste | Evaluación observada | Compras en la separación | Etiquetas inmaduras en ajuste | Etiquetas inmaduras en evaluación | Mediana de duración de las inmaduras de evaluación |
|---|---:|---:|---:|---:|---:|---:|
| Febrero | 43.609 | 6.231 | 7.069 | 84 | 324 | 45,95 d |
| Abril | 56.964 | 6.715 | 7.003 | 353 | 83 | 48,27 d |
| Junio | 71.023 | 6.070 | 6.749 | 95 | 26 | 55,96 d |
| Agosto | 83.933 | 6.351 | 6.156 | 30 | 0 | — |

Las inmaduras de ajuste duraron típicamente entre 48 y 59 días (medianas por corte), frente a la mediana global de 10,22. En febrero se excluyen 324 de 6.555 entregas finalmente observadas. La censura recorta desproporcionadamente la cola y puede favorecer modelos que subestimen duraciones extremas. Aunque agosto no censura casos elegibles de esa cohorte según este cierre, solo incluye 11 entregas de más de 30 días y ninguna de más de 60; no acredita buen rendimiento en la cola.

## Resultados

Todos los errores están en días; R² puede ser negativo cuando el error cuadrático supera al de una constante igual a la media del **conjunto evaluado** (referencia matemática que no se despliega). Las tolerancias de dos y cinco días son inclusivas. El [JSON de resultados](resultados_tiempo_entrega/metricas.json) incluye valores sin redondear, todos los cortes y bandas, sesgos, tiempos, versiones y hashes.

### Desarrollo: promedio de febrero, abril y junio

| Modelo o referencia | MAE | RMSE | R² | MedAE | ≤2 d | ≤5 d |
|---|---:|---:|---:|---:|---:|---:|
| Olist (timestamp exacto) | 14,20 | 15,83 | -5,509 | 13,92 | 3,48 % | 8,87 % |
| Olist (calendario) | 14,76 | 16,36 | -5,924 | 14,56 | 3,32 % | 8,30 % |
| Dummy mediana | 5,77 | 7,55 | -0,135 | 4,64 | 20,61 % | 53,36 % |
| Ridge | 5,39 | 7,01 | -0,048 | 4,26 | 23,51 % | 57,37 % |
| Random Forest | 4,93 | 6,54 | 0,125 | 3,87 | 25,68 % | 62,83 % |
| HGB | 4,76 | 6,49 | 0,166 | 3,60 | 29,41 % | 66,10 % |
| HGB + derivadas | 4,81 | 6,52 | 0,157 | 3,65 | 28,75 % | 65,54 % |
| **HGB sin plazo prometido** | **4,60** | **6,33** | **0,210** | **3,45** | **30,96 %** | **68,08 %** |
| XGBoost | 4,75 | 6,47 | 0,165 | 3,56 | 29,08 % | 66,58 % |

El ganador por MAE y RMSE macro fue **HistGradientBoostingRegressor sin plazo prometido**. Superó al Dummy y a ambas referencias de Olist en cada corte (MAE 6,05 / 4,20 / 3,55). Su ventaja macro sobre XGBoost es de 0,15 días y no domina todos los meses: Random Forest ganó en febrero (5,61) y XGBoost en abril (4,07). Las derivadas no mejoraron HGB en promedio. No se concluye que quitar la promesa sea siempre mejor: su valor predictivo varía con el período.

### Agosto retrospectivo: 6.351 pedidos

| Modelo o referencia | MAE | RMSE | R² | MedAE | ≤2 d | ≤5 d |
|---|---:|---:|---:|---:|---:|---:|
| Olist (timestamp exacto) | 8,14 | 10,94 | -5,359 | 6,15 | 18,12 % | 38,89 % |
| Olist (calendario) | 8,65 | 11,38 | -5,881 | 6,75 | 16,11 % | 36,92 % |
| Dummy mediana | 4,59 | 5,40 | -0,553 | 4,50 | 20,15 % | 58,70 % |
| Ridge | 4,17 | 5,29 | -0,486 | 3,41 | 29,35 % | 67,55 % |
| Random Forest | 3,40 | 4,51 | -0,080 | 2,63 | 39,96 % | 75,36 % |
| HGB | 2,92 | 4,10 | 0,105 | 2,13 | 47,72 % | 82,73 % |
| HGB + derivadas | **2,91** | **4,09** | **0,113** | 2,16 | 47,74 % | **82,95 %** |
| **HGB sin plazo, seleccionado en desarrollo** | **3,31** | **4,33** | **0,005** | **2,73** | **38,53 %** | **77,72 %** |
| XGBoost | 3,06 | 4,18 | 0,071 | 2,34 | 44,29 % | 80,93 % |

La alternativa seleccionada redujo el MAE en **4,82 días (59 %)** frente a Olist exacto, **5,33 días (62 %)** frente al calendario y **1,28 días (28 %)** frente a la mediana de entrenamiento. En promedio predice **1,58 días de más**; la promesa exacta es **7,45 días posterior** a la entrega real en promedio. El R² del ganador preseleccionado queda cerca de cero en agosto: mejora referencias que operan sin conocer ese mes, pero apenas supera en error cuadrático a una constante igual a la media del propio agosto.

El menor MAE de agosto se observó en HGB con derivadas (2,91), a solo 0,01 días del HGB estándar. XGBoost también superó al congelado en agosto, pero no en desarrollo. **No se cambia de ganador usando agosto.** El bootstrap descriptivo pareado por día (29 días, 500 réplicas) arroja para `MAE candidato − MAE comparador` los intervalos percentiles de 95 %: frente a Olist exacto **[-5,31; -4,37]**, Dummy **[-1,54; -1,05]** y HGB estándar **[+0,33; +0,45]** días. Esto resume variación dentro de un solo mes consultado; no incorpora la incertidumbre de seleccionar/entrenar ni dependencia entre días.

### Entregas rápidas, cola e interpretación

| Duración real | Pedidos de agosto | MAE candidato | MAE Olist exacto | MAE Dummy | Sesgo candidato (predicción − real) |
|---|---:|---:|---:|---:|---:|
| Hasta 5 días | 1.641 | 2,83 | 5,76 | 7,62 | +2,74 |
| >5 a 10 | 3.194 | 3,20 | 8,76 | 3,79 | +2,43 |
| >10 a 15 | 1.137 | 3,20 | 9,50 | 1,30 | +0,04 |
| >15 a 30 | 368 | 6,12 | 8,70 | 7,56 | -5,45 |
| >30 | 11 | **25,45** | **20,42** | 26,97 | **-25,45** |

La mejora promedio no se extiende a toda la distribución: el Dummy gana en 10–15 días y la promesa gana al candidato sobre más de 30 días. El candidato tiende a sobreestimar las entregas rápidas y a subestimar las largas. Estas bandas se conocen **después** de entregar; son diagnósticos, no reglas de predicción.

Importancia por permutación en **junio de desarrollo**, cinco repeticiones, medida como incremento del MAE: mismo estado (+0,41 días), flete (+0,38) y estado de cliente (+0,17). No implica causalidad. Permutar mes dentro de junio no lo altera; la promesa tiene importancia cero por diseño en ese modelo. Permutar estados independientemente de `mismo_estado` puede producir combinaciones que no ocurren en un pedido real; interpretar el orden de variables como orientativo.

### Coste de cómputo y artefacto

En el ajuste para agosto y en esta máquina, los tiempos aproximados de ajuste fueron Ridge 0,21 s, Random Forest 14,0 s, HGB estándar 5,0 s, HGB derivado 10,3 s, HGB sin plazo 5,3 s y XGBoost 3,0 s. En lotes de 1.024 predicciones (mediana de cinco medidas, incluye preprocesamiento, sin red): 6,4 / 30,2 / 12,8 / 25,8 / 13,8 / 6,2 ms, respectivamente. No son latencias de la API ni garantías para otro hardware. XGBoost fue más rápido en este ensayo, pero no mejor en el criterio de desarrollo; Random Forest no justificó su coste adicional. El HGB elegido tiene la ventaja de no necesitar XGBoost ni la promesa como entrada.

El candidato se guardó en `backend/app/models/delivery_regression_experimental.joblib` (326.117 bytes; SHA-256 `4a698a3771aa50c0e945d24db434fa7b074b3a16823201128966ba54d804df20`). Tras recargarlo, sus predicciones de agosto coincidieron dentro de `1e-10` días. Es un pipeline regresor local de scikit-learn; **no** es compatible como sustitución directa del clasificador que la API carga y consulta mediante `predict_proba`. No se activó ni se sobrescribieron artefactos previos. El binario está ignorado por Git; el script y el JSON permiten regenerarlo.

## Gráficos

- [Valores reales vs. predicción](resultados_tiempo_entrega/real_vs_prediccion.png)
- [Distribución de errores firmados](resultados_tiempo_entrega/distribucion_errores.png)
- [Errores por intervalo de duración real](resultados_tiempo_entrega/errores_por_duracion.png)
- [Comparación de modelos y referencias](resultados_tiempo_entrega/comparacion_modelos.png)
- [Importancia por permutación en desarrollo](resultados_tiempo_entrega/importancia_permutacion.png)

## ¿Es más conveniente que la clasificación de retrasos?

**Sí como candidato experimental para estimar cuándo llegará un pedido; no como reemplazo inmediato de la clasificación.** La regresión da información utilizable para planificación en días, aun cuando la promesa no está disponible. Ganó en desarrollo y reduce sustancialmente el error promedio frente a la promesa de Olist como predicción puntual. Pero no cuantifica el riesgo de incumplimiento, subestima la cola y todavía no tiene una prueba externa madura e independiente.

Como diagnóstico secundario, sin optimizar una regla sobre agosto, se marca tardío si `compra + duración_predicha` alcanza la medianoche posterior al día prometido. La matriz es `[[5559,399],[300,93]]` (`[[TN,FP],[FN,TP]]`), precisión 18,90 % y recall 23,66 %. El HGB clasificador anterior, con su propio umbral de validación, detectó 114 y omitió 279 con 831 falsas alertas. La regla por ETA genera menos falsas alertas pero detecta menos atrasos; no es demostración de un mejor detector. MAE/RMSE de días y precisión/recall de alertas responden preguntas distintas y no se comparan directamente.

**Recomendación:** mantener intacto el servicio actual y llevar el HGB sin promesa a una nueva validación de ETA, junto con el HGB estándar como referencia de sensibilidad temporal. Antes de mostrar un plazo a clientes, medir de nuevo en datos recientes y maduros, definir tolerancias de negocio, y evaluar intervalos o cuantiles y su cobertura por ruta y duración. Si la necesidad principal sigue siendo alertar retrasos, conservar esa decisión como un problema separado. No está listo para producción.

## Reproducción

Desde la raíz de la copia principal:

```powershell
$xgb = 'C:\Users\MSI\AppData\Local\Temp\opencode\delivery-regression-xgb'
.\backend\.venv\Scripts\python.exe -B -m pip install matplotlib==3.11.2
.\backend\.venv\Scripts\python.exe -B -m pip install --only-binary=:all: --no-deps --target $xgb xgboost==3.4.1
.\backend\.venv\Scripts\python.exe -B -m ml.compare_delivery_regression --require-xgboost --xgboost-path $xgb
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s ml/tests
```

XGBoost se instaló en un directorio temporal local; matplotlib solo en la `.venv` del proyecto. No se modificaron herramientas globales ni requisitos versionados. Si XGBoost no está disponible, se puede omitir `--require-xgboost`; el JSON registrará su ausencia. Para repetir, usar rutas nuevas con `--output` y `--artifact`: el script no sobrescribe. No ejecutar `ml/train_delay.py` sin `--check-data` para esta evaluación: es otro experimento y usa el nombre de modelo activo para clasificación.

**Pruebas: 63 aprobadas** (54 previas y 9 nuevas). Se verificaron hashes de los seis CSV, del código experimental y de los artefactos previos. No hubo cambios en frontend, endpoints, contratos, otros módulos ni copia duplicada; sin commits ni push.
