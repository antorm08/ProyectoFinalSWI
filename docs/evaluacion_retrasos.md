# Evaluación exploratoria de retrasos logísticos

**Ejecución:** 23 de noviembre de 2026, 00:46–00:47 UTC (22 de noviembre en horario local).  
**Dictamen:** señal predictiva académicamente útil, pero rendimiento operativo limitado y **probabilidades no calibradas**. No listo para producción.

Resultados íntegros, curvas de umbrales, bandas de calibración, versiones y hashes: [evaluacion_retrasos_metricas.json](evaluacion_retrasos_metricas.json). Contexto: [preparacion_olist.md](preparacion_olist.md) y [validacion_olist.md](validacion_olist.md), conservados como antecedentes históricos.

## 1. Configuración y prevención de fuga

Se leyeron íntegramente los dos informes y `ml/common.py`, `ml/temporal.py`, `ml/train_delay.py` y el servicio de retrasos antes de entrenar. Las **40 pruebas de preparación pasaron**.

El experimento reproducible está en `ml/evaluate_delay.py`; reutiliza sin modificar la preparación ni el constructor del modelo existente.

- Objetivo: entrega en un **día calendario posterior** al prometido.
- Universo: 96.470 pedidos elegibles; 6.534 tardíos. Ocho entregados sin fechas de objetivo siguen sin ser etiquetables.
- Entradas exclusivas: `precio`, `flete`, `peso_g`, `n_items`, `dias_estimados`, `mes`, `dia_semana`, `mismo_estado`, `estado_cliente`, `estado_vendedor`.
- Sin identificadores, reseñas, duración real ni fechas de entrega como predictores. Las fechas solo particionan/auditan; `mes` y `dia_semana` provienen de compra.
- Pipeline: OneHotEncoder ajustado solo con entrenamiento, descarte explícito del resto de columnas, HistGradientBoostingClassifier con NaN nativo.
- Configuración principal: 300 iteraciones, learning rate 0,1, 31 hojas máximas, mínimo 20 observaciones por hoja, `class_weight="balanced"`, `early_stopping=False`, semilla 42. Sin búsqueda de hiperparámetros, sobremuestreo ni refit con validación/prueba.
- Un hilo numérico; Python 3.12.10, scikit-learn 1.9.1, pandas 3.1.1, NumPy 2.5.5, SciPy 1.18.1, joblib 1.6.0. No hay garantía binaria entre versiones distintas.
- Baseline: `DummyClassifier(strategy="prior")`, ajustado **solo** a la prevalencia de entrenamiento, no a la del conjunto evaluado. Es una referencia probabilística de ausencia de información; también se muestran sus decisiones mayoritarias y su umbral validado.

El modelo principal se ajustó con entrenamiento exclusivamente. Los dos ajustes de sensibilidad, tres ventanas anteriores y una repetición de reproducibilidad son experimentos separados: ninguno reemplazó al candidato ni se ajustó con prueba.

## 2. Particiones y disponibilidad de etiquetas

Los límites de compra son inclusivos al inicio y exclusivos al final; los cortes de disponibilidad son estrictos.

| Conjunto | Compra | Entrega real anterior a | Pedidos | Tardíos | Prevalencia |
|---|---|---|---:|---:|---:|
| Entrenamiento | Antes de 2018-05-01 | 2018-06-01 | 71.023 | 5.331 | 7,506 % |
| Validación | Junio de 2018 | 2018-08-01 | 6.070 | 51 | 0,840 % |
| Prueba final | Agosto de 2018 | 2018-10-18 | 6.351 | 393 | 6,188 % |

Últimas etiquetas: entrenamiento **2018-05-31 00:09:39**, validación **2018-07-31 20:36:42**, prueba **2018-09-25 00:47:25**. Las verificaciones de disponibilidad pasaron.

Se conservan aparte 6.749 compras de mayo y 6.156 de julio como intervalos de maduración. Además, 95 compras de entrenamiento y 26 de validación no tenían etiqueta disponible en su corte. La suma de todos los grupos es 96.470 y no hay pedidos compartidos.

La prevalencia de entrenamiento es aproximadamente nueve veces la de validación; la de prueba es más de siete veces la de validación. No procede asumir estacionariedad ni interpretar el buen ordenamiento relativo en junio como precisión estable en agosto. El cambio combina evolución temporal y selección por entregas observables; estos datos no permiten separar causalmente sus contribuciones.

### Sesgo de entregas largas

La entrega real es un **proxy** de disponibilidad: no hay timestamps de ingestión, historial de estados ni garantía de que el evento se conociera inmediatamente.

| Excluidos por etiqueta inmadura | Pedidos | Tardíos | Duración media | Máxima |
|---|---:|---:|---:|---:|
| Entrenamiento | 95 | 88 | 60,78 días | 208,35 días |
| Validación | 26 | 20 | 57,77 días | 136,78 días |

La exclusión es informativa, no aleatoria: concentra retrasos difíciles. Junio completo tiene 6.096 pedidos y 71 tardíos (1,165 %); al corte solo se observan 51. El intervalo de maduración **reduce, pero no elimina**, la censura. La población ya está condicionada a pedidos finalmente entregados; no cubre cancelaciones.

Tras congelar todas las decisiones y evaluar prueba, se inspeccionaron las etiquetas futuras únicamente como diagnóstico: en los 26 casos omitidos de validación el modelo detecta 2 de 20 tardíos; en junio completo obtiene precision 0,0826, recall 0,1408 y F1 0,1042. Estos resultados **no se usaron** para escoger umbral, calibrador o modelo.

## 3. Umbral elegido exclusivamente en validación

Se evaluó una grilla fija de **0,01 a 0,99**, paso 0,01. Criterio: F1 de la clase tardía máximo; desempate por precision y luego umbral mayor. No existe una matriz de costes de negocio que permita afirmar optimalidad económica.

| Umbral HGB | Precision validación | Recall | F1 | Alertas |
|---|---:|---:|---:|---:|
| 0,50, referencia predefinida | 0,0533 | 0,1569 | 0,0796 | 150 |
| **0,53, seleccionado** | **0,0672** | **0,1569** | **0,0941** | **119** |

El umbral y el candidato quedaron congelados a **00:47:41.052190 UTC**, antes del scoring final registrado en el JSON. La grilla completa se conserva allí. Maximizar sobre solo 51 positivos puede sobreajustar el umbral: la métrica de validación elegida tiene optimismo de selección.

Para el baseline, la misma regla seleccionó 0,07: como su score constante es 0,07506, predice todos tardíos. A 0,50 predice todos puntuales. Es un comportamiento degenerado transparente, no una solución logística.

**El umbral experimental 0,53 no se incorporó al servicio ni al pipeline.** `predict` del clasificador conserva su regla predeterminada; la decisión experimental se calcula externamente con `predict_proba >= 0.53`. El servicio mantiene bandas fijas 0,20/0,50, que no equivalen a este clasificador binario.

## 4. Métricas reales

Métricas de clase positiva. Entrenamiento es evaluación **aparente**, no una estimación de generalización. AP es average precision (resumen PR no interpolado); se muestra también AUC trapezoidal para evitar ambigüedad con «PR-AUC».

### HGB con umbral validado 0,53

| Conjunto | Precision | Recall | F1 | ROC-AUC | PR-AUC (AP) | PR trapezoidal | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Entrenamiento aparente | 0,2836 | 0,8060 | 0,4195 | 0,9049 | 0,4353 | 0,4351 | 0,13665 |
| Validación | 0,0672 | 0,1569 | 0,0941 | 0,7698 | 0,0392 | 0,0359 | 0,05159 |
| **Prueba definitiva** | **0,1206** | **0,2901** | **0,1704** | **0,6632** | **0,1087** | **0,1075** | **0,14209** |

ROC-AUC, AP y Brier no dependen de este umbral. La brecha entrenamiento/futuro evidencia generalización limitada, compatible con sobreajuste y cambio temporal; no permite atribuirla solo a uno de ellos.

### Baseline con su umbral validado 0,07

| Conjunto | Precision | Recall | F1 | ROC-AUC | AP | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Entrenamiento aparente | 0,0751 | 1,0000 | 0,1396 | 0,5000 | 0,0751 | 0,06943 |
| Validación | 0,0084 | 1,0000 | 0,0167 | 0,5000 | 0,0084 | 0,01277 |
| Prueba | 0,0619 | 1,0000 | 0,1165 | 0,5000 | 0,0619 | 0,05822 |

A 0,50, precision/recall/F1 del baseline son cero en los tres conjuntos; sus métricas probabilísticas no cambian. HGB mejora ranking y F1 frente a estas referencias, pero **pierde claramente en Brier y log-loss**: en prueba log-loss 0,44556 frente a 0,23343 del baseline.

En prueba, AP del HGB es aproximadamente 1,76 veces la prevalencia: hay señal, no un detector fiable. La AUC PR trapezoidal del baseline constante es 0,53094 por interpolación entre pocos puntos; **no significa rendimiento útil**. AP=0,06188 es la referencia PR apropiada para comparar ese baseline.

### Matrices de confusión

Filas: real puntual/tardío. Columnas: predicho puntual/tardío; formato `[[TN, FP], [FN, TP]]`.

| Modelo y regla | Entrenamiento | Validación | Prueba |
|---|---|---|---|
| HGB, 0,53 | `[[54836,10856],[1034,4297]]` | `[[5908,111],[43,8]]` | **`[[5127,831],[279,114]]`** |
| HGB, 0,50 | `[[53567,12125],[901,4430]]` | `[[5877,142],[43,8]]` | `[[4993,965],[260,133]]` |
| Baseline, 0,07 | `[[0,65692],[0,5331]]` | `[[0,6019],[0,51]]` | `[[0,5958],[0,393]]` |
| Baseline, 0,50 | `[[65692,0],[5331,0]]` | `[[6019,0],[51,0]]` | `[[5958,0],[393,0]]` |

A 0,50 el HGB da en prueba precision 0,1211, recall 0,3384 y F1 0,1784. Aunque supera el F1 de 0,53 en agosto, **no se cambia el umbral retrospectivamente**.

## 5. Calidad de probabilidades y calibración

Las salidas del clasificador ponderado **no representan directamente riesgos**:

| Conjunto | Score medio HGB | Frecuencia real | ECE, 10 bandas uniformes |
|---|---:|---:|---:|
| Entrenamiento aparente | 33,52 % | 7,51 % | No calculado |
| Validación | 17,96 % | 0,84 % | 0,17122 |
| Prueba | 31,05 % | 6,19 % | 0,24866 |

Ejemplos de fiabilidad en prueba: banda [0,50;0,60), 479 pedidos, score medio 0,548 y frecuencia 0,113; banda [0,70;0,80), 181 pedidos, score medio 0,740 y frecuencia 0,127. El JSON conserva todas las bandas, incluidas vacías. ECE depende de su discretización y no reemplaza Brier ni una evaluación temporal de calibración.

Se diagnosticó en validación la corrección algebraica de odds para deshacer pesos balanceados, usando **solo** la prevalencia de entrenamiento:

```text
r = (1 - prevalencia_train) / prevalencia_train
p_diagnostico = score / (score + r * (1 - score))
```

Reduce Brier de 0,05159 a 0,00849 y score medio a 0,02029, aún por encima de 0,00840 observado. Es una transformación bajo supuestos ideales, **no calibración demostrada**, y no resuelve el cambio de prevalencia. No se seleccionó ni persistió esa transformación ni se compararon calibradores sobre prueba.

**Decisión:** se necesita calibración antes de exponer porcentajes como riesgo. No se ajustó Platt/isotónica sobre los 51 positivos censurados de junio, ni con la prueba final. Se necesitaría un período de calibración independiente, suficientemente maduro y más representativo, y después otra evaluación temporal intacta. El candidato guardado conserva scores crudos y permanece inactivo.

## 6. Sensibilidad y estabilidad temporal

### Calidad de entrenamiento

Anomalía auxiliar: transportista anterior a compra/aprobación, o entrega al cliente anterior a aprobación/transportista. Fechas ausentes no se inventan como anomalías de orden. Las fechas imposibles del objetivo ya son rechazadas por preparación.

Variantes ajustadas solo con subconjuntos de entrenamiento, evaluadas con **la misma validación y umbral 0,53**, sin reemplazar al principal:

| Variante | Entrenamiento (tardíos) | ROC-AUC validación | AP | F1 | Brier |
|---|---:|---:|---:|---:|---:|
| Principal | 71.023 (5.331) | 0,7698 | 0,0392 | 0,0941 | 0,05159 |
| Sin anomalías auxiliares | 70.490 (5.320) | 0,7771 | 0,0452 | 0,0930 | 0,05187 |
| Sin entregas >60 días | 70.769 (5.081) | 0,7511 | 0,0431 | 0,0952 | 0,05024 |

Eliminar las 533 anomalías de entrenamiento (11 tardías) no mejora el F1; no justifica descartarlas como política general.

Las entregas >60 días excluidas son 254, de ellas 250 tardías: filtrarlas introduce selección por resultado futuro y **no** se propone como regla de inferencia. Ninguna sensibilidad prueba que borrar casos difíciles mejore generalización.

### Tres ventanas anteriores, sin agosto

Mismos hiperparámetros, umbral **predefinido 0,50**, sin seleccionar por sus resultados. Cada ajuste deja un mes entre fin de compras de entrenamiento e inicio de evaluación y otro para madurar las etiquetas evaluadas.

| Compras entrenamiento antes de | Etiquetas train antes de / mes evaluado | Corte etiquetas evaluación | N train | N evaluación (tardíos) | ROC-AUC | AP | F1 |
|---|---|---|---:|---:|---:|---:|---:|
| 2017-11-01 | 2017-12-01 / diciembre | 2018-02-01 | 30.852 | 5.444 (347) | 0,6347 | 0,1124 | 0,1224 |
| 2017-12-01 | 2018-01-01 / enero | 2018-03-01 | 37.865 | 6.890 (259) | 0,5609 | 0,0419 | 0,0533 |
| 2018-01-01 | 2018-02-01 / febrero | 2018-04-01 | 43.609 | 6.231 (610) | 0,6291 | 0,1568 | 0,2155 |

Las prevalencias evaluadas son 6,374 %, 3,759 % y 9,790 %; quedan 69, 179 y 324 etiquetas inmaduras respectivamente. La variabilidad confirma fragilidad temporal; no se promediaron estas ventanas como si fueran observaciones independientes ni se eligió la mejor.

## 7. Errores e interpretación

- En prueba, el umbral elegido genera **945 alertas: 114 aciertos y 831 falsas alarmas**. Omite **279 de 393 retrasos**. Aproximadamente 88 % de las alertas son falsas.
- El umbral histórico del servicio 0,20 marcaría 4.270 de 6.351 pedidos: recall 0,8524 pero precision 0,0785 y 3.935 falsas alarmas. Es un diagnóstico de una regla ya existente, no un umbral elegido con prueba.
- Entre 11 entregas de más de 30 días en prueba, todas tardías, detecta solo una. No hay casos >60 días en esa ventana: no se puede afirmar buen comportamiento sobre extremos. AP=1 en un subgrupo solo positivo no acredita discriminación; ROC-AUC es indefinida.
- Entre estados representativos distintos: 3.511 pedidos, 105 tardíos; solo detecta 4, con 239 falsas alarmas (recall 0,0381, precision 0,0165). En mismo estado: recall 0,3819 y precision 0,1567. Son asociaciones descriptivas posteriores, no causalidad ni reglas para reoptimizar.
- Hay 86 anomalías auxiliares en prueba con solo dos tardíos; los subgrupos de peso desconocido (4) y multiorigen (55) no tienen positivos. No sustentan conclusiones de robustez.

Importancia por permutación de **validación**, caída media de AP en tres repeticiones: estado cliente 0,02341, días prometidos 0,01760, flete 0,01373, peso 0,00742 y estado vendedor 0,00437. No se reajustaron variables con esto. Las correlaciones, los pocos positivos y el mes constante en la ventana limitan la interpretación; importancia no implica causalidad ni explica individualmente cada error.

## 8. Artefacto, compatibilidad y reproducibilidad

Guardado únicamente tras pasar verificaciones:

- Ruta: `backend/app/models/delay_model_exploratorio.joblib`, **449.874 bytes**, pipeline sklearn nativo.
- SHA-256: `793b4a82e037d2541f5e02faafa0b7932a2e3f48fba2763562cd475ef096f315`.
- Dos ajustes independientes con la misma configuración: diferencia máxima de scores **0,0** en toda validación y 1.024 casos de entrenamiento.
- Serialización/carga: scores exactamente iguales en toda validación.
- Servicio real y ruta ASGI `POST /api/delay/predict`: nueve casos con HTTP 200 y respuesta igual al pipeline redondeado, incluyendo pesos desconocidos/cero, estados en minúsculas, multiorigen y multiartículo; entrada negativa rechazada con HTTP 422.
- Solo se sustituyeron temporalmente en memoria la ruta del artefacto y la caché durante esas comprobaciones. No se modificaron configuración, servicio, endpoint ni esquema HTTP. Es una prueba en proceso, no un despliegue ni una prueba de red.
- **No se creó `delay_model.joblib` ni se activó el candidato.** Las bandas del servicio seguirían tratando scores crudos como porcentajes si se activara sin una decisión posterior; por ello no se recomienda hacerlo.
- Los seis CSV y los artefactos preexistentes mantienen sus hashes. El artefacto nuevo está ignorado por Git según las reglas existentes; conservarlo aparte junto con su JSON si se requiere entregar el binario.

Reproducción:

```powershell
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s ml/tests -v
.\backend\.venv\Scripts\python.exe -B -m ml.evaluate_delay
```

El segundo comando se ejecutó una vez para este experimento y **rechaza sobrescribir** modelo o JSON existentes. Para repetir, usar `--model-path` y `--metrics-path` con nombres nuevos dentro del repositorio, sin borrar el experimento original. Repetirlo no convierte agosto de nuevo en un conjunto desconocido.

Se añadieron ocho pruebas de métricas, degeneración del baseline, umbrales, fiabilidad y protección contra sobrescritura, sin entrenamiento en esas pruebas. Se preservaron las 40 pruebas previas.

**Verificación final: 48 pruebas aprobadas**, en 17,752 segundos. `git diff --check` sin errores. Después de la batería se comprobaron de nuevo los hashes de los seis CSV, del código experimental y del candidato; todos coinciden con el registro del experimento. El experimento completo duró 56,67 segundos. El script utiliza un directorio temporal local de OpenCode para verificar serialización; esa ruta debe existir al reproducirlo en este entorno.

## 9. Limitaciones y viabilidad académica

La prueba final ya fue observada: mejoras posteriores deben reservar otro período externo o declarar explícitamente la reutilización exploratoria. No se ofrecen intervalos de confianza, significación estadística ni garantías de estabilidad por vendedor/comprador; las dependencias entre pedidos impiden tratar sin más todas las filas como independientes.

Persisten la censura informativa, las anomalías auxiliares, la población condicionada a entrega final, el catálogo sin historial, el origen único aproximado y el cambio temporal. El modelo no atribuye responsabilidades a vendedores ni predice cancelaciones. No se han definido costes de intervención.

**Conclusión:** viable como experimento académico reproducible que demuestra tanto señal sobre el baseline como fallos claros de generalización y calibración. No permite presentar un predictor fiable de riesgo individual ni una solución lista para producción. Antes de cualquier activación hacen falta calibración temporal independiente, datos más maduros y representativos, revisión de la elevada carga de falsas alertas y nueva validación externa.

Sin cambios en frontend, CSV, otros módulos de IA ni contratos HTTP. Sin commits ni push.
