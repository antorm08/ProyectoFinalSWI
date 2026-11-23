# Segunda evaluación experimental de retrasos

Este estudio compara candidatos del módulo 1 de CommerceAI sin cambiar el objetivo (entrega posterior al día calendario prometido), los CSV ni el contrato HTTP. Complementa [la primera evaluación](evaluacion_retrasos.md) y [la preparación de Olist](preparacion_olist.md).

**Agosto de 2018 no es una prueba externa inédita:** sus resultados anteriores ya eran públicos. En esta ronda no entra en selección, ajuste, calibración ni selección de umbral. Solo se vuelve a puntuar después de fijar el candidato, sobre los mismos pedidos, para una comparación retrospectiva pareada. La confirmación de mejoras requeriría otra cohorte.

## Diagnóstico de partida

El HGB anterior, ponderado por clase, detectó 114 de 393 atrasos en agosto con 831 falsas alarmas y 279 omisiones. Solo detectó 4 de 105 retrasos entre estados representativos diferentes y 1 de 11 entregas de más de 30 días. Su score medio fue 31,05 % frente al 6,19 % observado; su Brier (0,1421) fue peor que el de un prior constante (0,0582). El desbalance y la deriva son distintos: en el estudio anterior, entrenamiento, junio disponible y agosto tenían 7,51 %, 0,84 % y 6,19 % de positivos. Junio completo habría tenido 1,16 %: el corte agrava la caída, pero no explica toda la diferencia. El score ponderado no debe leerse como riesgo.

## Protocolo

Cada corte tiene tres bloques de compras disjuntos: ajuste base, calibración y evaluación. Se deja un mes de maduración entre bloques. Para entrenar se exige entrega real estrictamente anterior al inicio de calibración; para calibrar, anterior al inicio de evaluación; para evaluar, anterior al cierre observacional del corte. El estimador queda congelado al ajustar el calibrador. El preprocesamiento se aprende solo en ajuste. Los pliegues son de origen rodante: un pedido puede aparecer en distintos roles en pliegues diferentes, pero nunca en dos roles dentro del mismo pliegue. Los bloques evaluados no se solapan.

| Corte | Compras para ajuste base, antes de | Compras para calibración | Compras para evaluación | Cierre de etiquetas evaluadas (exclusivo) |
|---|---|---|---|---|
| D1 | 2017-09-01 | oct–nov 2017 | ene 2018 | 2018-03-01 |
| D2 | 2017-11-01 | dic 2017–ene 2018 | mar 2018 | 2018-05-01 |
| D3 | 2018-01-01 | feb–mar 2018 | may 2018 | 2018-07-01 |
| Comparación congelada | 2018-04-01 | may–jun 2018 | ago 2018 | 2018-10-18 |

Entrega real es solo un proxy de disponibilidad; no se conoce el timestamp de ingestión. Se registran las etiquetas inmaduras, sin incorporarlas ni imputarlas. Su exclusión no es aleatoria y el estudio sigue condicionado a pedidos finalmente entregados. No se usan fechas de aprobación, transporte, entrega ni reseñas como predictores.

### Variantes y variables

- **Regresión logística** con y sin pesos balanceados, `C=1`, one-hot de estados, mediana (con indicador de peso desconocido si existe en ajuste) y escalado numérico aprendidos por pliegue.
- **HGB original** (300 iteraciones, hasta 31 hojas, mínimo 20 observaciones por hoja y tasa 0,1) con y sin balanceo; ambos comparables en igualdad de hiperparámetros.
- **HGB compacto** (150 iteraciones, hasta 15 hojas, mínimo 80 por hoja, tasa 0,07 y regularización L2 de 2) sin pesos, con peso tardío 3 y con balanceo inverso de clases. Sin sobremuestreo ni parada temprana aleatoria. Se comparan también dos variantes compactas con variables derivadas, con pesos 1 y 3 respectivamente.
- **Cinco derivadas disponibles sin ampliar el contrato:** `flete/(precio+flete+0,01)`, precio por artículo, peso por artículo (conserva NaN), flete cero y compra en fin de semana. Son transformaciones de los campos ya conocidos al comprar; no aportan datos externos. Hora de compra, volumen de catálogo, rutas completas y recuento real de vendedores no se incorporan: faltan en el contrato; las dimensiones además carecen de historial verificable. Información histórica de vendedor exigiría reconstrucción temporal y datos de inferencia no disponibles.
- **Probabilidades:** se comparan scores crudos, calibración sigmoide e isotónica con sklearn y estimador base congelado. Los calibradores no usan ponderación de clases ni etiquetas de evaluación. Se selecciona por Brier; isotónica puede introducir empates y perder resolución de ranking. El cambio de prevalencia puede perjudicar también a un modelo calibrado.

Se elige la familia por mayor **AP macro** (promedio no ponderado de los tres meses en crudo), el método de calibración por menor **Brier macro** de ese ganador, y el umbral por máximo F1 positivo sobre predicciones de desarrollo concatenadas, con a lo sumo **10 % de alertas agrupadas**, sin cortar empates. El 10 % es una restricción experimental ilustrativa, no capacidad acordada ni garantía por mes. Se reajustan únicamente la variante y el método elegidos en los bloques finales de ajuste y calibración; el umbral no se vuelve a estimar. Los cortes y las variantes están en [`ml/improve_delay.py`](../ml/improve_delay.py).

Las métricas por corte se conservan en [el JSON experimental](mejora_retrasos_metricas.json). AP es el resumen no interpolado de precision–recall (PR-AUC en este informe); el JSON conserva además el área trapezoidal. El Brier se calcula sobre probabilidades, no sobre decisiones. La comparación usa bootstrap pareado por día de compra (500 réplicas, semilla 42): conserva la dependencia intradía y empareja los mismos casos. Los percentiles describen variación **dentro de agosto**, no incorporan selección/entrenamiento ni eliminan dependencias entre días, p. ej. por vendedor; no son una validación externa ni una prueba confirmatoria.

### Reproducción

```powershell
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s ml/tests
.\backend\.venv\Scripts\python.exe -B -m ml.improve_delay
```

La suite pasó **54 pruebas** (48 previas y 6 nuevas). El experimento limita el cálculo a un hilo y registra versiones, hashes de entradas y código, las ventanas y el momento de congelar la selección. Para repetir con resultados existentes, proporcionar rutas nuevas mediante `--output` y `--artifact`. El script rechaza destinos existentes y la ruta activa del servicio.

Hay que distinguir la comparación entre variantes en desarrollo, con idénticos cortes, de la comparación final entre artefactos: esta última contrasta dos protocolos completos, con distintos períodos de ajuste y reglas de decisión. No permite atribuir una diferencia exclusivamente a la calibración, al peso de clase o al algoritmo.

## Resultados de desarrollo

Cada celda de los tres bloques muestra pedidos / retrasos observables. La última columna cuenta las entregas no disponibles en el corte de evaluación (entre paréntesis, las que resultaron tardías retrospectivamente).

| Período de evaluación | Ajuste base | Calibración | Evaluación | Prevalencia observada | No disponibles (tardíos) |
|---|---:|---:|---:|---:|---:|
| Enero | 22.241 / 749 | 11.452 / 794 | 6.890 / 259 | 3,76 % | 179 (144) |
| Marzo | 30.852 / 1.105 | 12.392 / 659 | 6.816 / 1.152 | 16,90 % | 187 (176) |
| Mayo | 43.609 / 2.377 | 13.339 / 2.046 | 6.726 / 427 | 6,35 % | 23 (16) |

Las cohortes completas habrían tenido 5,70 %, 18,96 % y 6,56 % de retrasos respectivamente. Las diferencias persisten incluso sin ese filtro; entre ventanas de calibración la prevalencia ya variaba de 5,32 % a 15,34 %. El candidato final se ajustó con 64.094 pedidos, 4.898 tardíos (7,64 %), con última etiqueta el 30 de abril; quedaron fuera 226 aún inmaduros, 215 finalmente tardíos. Mayo–junio aporta el bloque de calibración reservado de 12.817 pedidos y 492 tardíos (3,84 %), última etiqueta el 31 de julio; 28 inmaduros, 22 finalmente tardíos. Como ganó la versión cruda, este bloque final **no se utilizó para ajustar ningún componente del artefacto**.

### Desbalance, regularización y variables derivadas

Promedios macro de los tres cortes, antes de fijar umbral. Para cada calibrador se muestra el Brier fuera de su propio bloque de ajuste; las métricas y matrices por corte están en el JSON.

| Modelo | AP cruda ↑ | ROC-AUC cruda ↑ | Brier crudo ↓ | Brier sigmoide ↓ | Brier isotónico ↓ |
|---|---:|---:|---:|---:|---:|
| **Logística sin pesos** | **0,14162** | **0,62561** | **0,08229** | 0,09196 | 0,09059 |
| Logística balanceada | 0,13981 | 0,61981 | 0,27777 | 0,09049 | 0,08962 |
| HGB original balanceado | 0,11722 | 0,57308 | 0,12295 | 0,08544 | 0,08577 |
| HGB original sin pesos | 0,11238 | 0,58084 | 0,08759 | 0,08563 | 0,08572 |
| HGB compacto sin pesos | 0,12162 | 0,58710 | 0,08286 | 0,08555 | 0,08560 |
| HGB compacto, peso 3 | 0,12176 | 0,58829 | 0,08446 | 0,08532 | 0,08561 |
| HGB compacto balanceado | 0,12765 | 0,59363 | 0,20819 | 0,08518 | 0,08543 |
| HGB compacto sin pesos + derivadas | 0,12495 | 0,59721 | 0,08276 | 0,08526 | 0,08530 |
| HGB compacto, peso 3 + derivadas | 0,12614 | 0,60125 | 0,08349 | 0,08527 | 0,08540 |

El balanceo elevó ligeramente AP en los HGB de igual estructura, a costa de empeorar Brier crudo; en logística empeoró ambas métricas. El peso 3 evitó gran parte del deterioro, pero no dio una mejora clara de AP frente a no ponderar. Las derivadas mejoraron AP entre 0,0033 y 0,0044 frente a sus pares sin derivadas, sin alcanzar a la logística. No se interpreta esta diferencia pequeña como una ganancia demostrada.

La recalibración deshizo gran parte del daño del balanceo, pero para la logística sin pesos **ninguno de los calibradores mejoró Brier macro**. Esto no acredita que su salida cruda esté calibrada: en marzo, por ejemplo, el score crudo medio fue 5,56 % para una frecuencia de 16,90 %. Una calibración aprendida en una cohorte previa no garantiza transportabilidad frente a esa deriva.

La logística cruda quedó congelada con **umbral 0,08925342821341102**. En desarrollo agrupado marcó 2.039/20.432 pedidos (9,98 %), precision 0,1672, recall 0,1855, F1 0,1759, ROC-AUC 0,6131, AP 0,1367 y Brier 0,08223; matriz `[[16896,1698],[1497,341]]`. Las tasas de alerta por mes fueron 13,64 %, 11,44 % y 4,74 %: aun dentro de desarrollo, el límite agregado no equivale a un límite mensual.

## Comparación retrospectiva en agosto

Selección congelada a las `03:27:50.507419 UTC`; primer scoring de agosto registrado a las `03:27:50.754758 UTC`. Son los mismos 6.351 pedidos y 393 retrasos (6,19 %) del informe original, con las etiquetas disponibles a su cierre. Se reprodujo exactamente la matriz histórica antes de comparar.

| Métrica | HGB anterior, umbral 0,53 | Logística cruda, umbral 0,08925 |
|---|---:|---:|
| ROC-AUC | **0,6632** | 0,5962 |
| PR-AUC (AP) | **0,1087** | 0,0812 |
| Precision | **0,1206** | 0,0831 |
| Recall | 0,2901 | **0,7659** |
| F1 | **0,1704** | 0,1499 |
| Brier ↓ | 0,14209 | **0,06341** |
| Alertas emitidas | 945 (14,88 %) | 3.624 (57,06 %) |
| Falsas alarmas | 831 | 3.323 |
| Retrasos omitidos | 279 | 92 |
| Matriz `[[TN,FP],[FN,TP]]` | `[[5127,831],[279,114]]` | `[[2635,3323],[92,301]]` |

Se detectan **187 retrasos netos adicionales a cambio de 2.492 falsas alarmas netas adicionales** (unas 13,3 por cada retraso adicional). Es una comparación de los dos umbrales fijados, no a igual presupuesto. El 57,06 % de alertas muestra que el 10 % de desarrollo no sirve como techo operativo. No se buscó otro umbral en agosto. Sin costes de intervención y de omisión, el mayor recall no justifica ese volumen.

El deterioro de ranking agregado convive con una mejora descriptiva de ROC-AUC en cada grupo geográfico por separado. En estados representativos distintos la logística detecta 95/105 retrasos vs. 4/105, pero produce 2.342 falsas alarmas vs. 239. Su score medio para pedidos interestatales es 13,70 % aunque el retraso observado es 2,99 %; dentro del mismo estado asigna 8,32 % frente a 10,14 % observado. Esta inversión entre grupos ayuda a entender por qué el ordenamiento agregado falla. Para entregas de más de 30 días detecta 8/11 vs. 1/11; son solo once positivos y no permiten evaluar discriminación ni concluir robustez en la cola.

**Calidad probabilística:** el score medio cae de 31,05 % a 11,30 %, aún por encima del 6,19 % real. El ECE (10 bandas) baja de 0,2487 a 0,0511. Sin embargo, el prior constante del ajuste final (7,64 %) obtiene Brier **0,05826** y log-loss **0,23370**, mejores que 0,06341 y 0,24787 de la logística. En la banda del 10–20 %, la logística predice 13,16 % en promedio y se observa 8,11 %. No hay evidencia de que sus probabilidades aporten una mejora frente a un pronóstico constante en este mes.

**Incertidumbre pareada descriptiva**, diferencias logística menos HGB (intervalos percentiles de 95 %, 29 días, 500 réplicas): ROC-AUC `[-0,0982; -0,0233]`, AP `[-0,0489; -0,0015]`, Brier `[-0,0893; -0,0695]`, recall `[+0,4348; +0,5217]`, F1 `[-0,0456; +0,0030]`. La mejora de Brier y recall respecto al artefacto anterior y la pérdida de ranking se sostienen dentro de este remuestreo; el intervalo de F1 incluye cero. Esto no prueba mejoras generalizables ni corrige el coste observado de alertas.

## Artefactos y recomendación

La logística cruda queda como **referencia experimental**, no como sustituto del HGB: se conserva en `backend/app/models/delay_model_mejora_experimental.joblib` (2.699 bytes, SHA-256 `6e086a77a83d36a2b50368eb5502c0d37aa3612d198bea87e18b00234dcbf2d8`). La recarga reproduce exactamente los scores comprobados; con la ruta sustituida solo durante la prueba, el servicio real coincidió con el pipeline en siete peticiones HTTP 200 y rechazó un valor inválido con HTTP 422. No se cambió el contrato ni las bandas fijas del servicio; el umbral experimental no forma parte de ellas. El artefacto previo permanece intacto y el nuevo **no se activó**.

**Recomendación para CommerceAI: no activar ninguno de los dos.** Usar el HGB anterior como referencia de ranking y la logística sin pesos como baseline interpretable para la próxima ronda. La mejora de recall y Brier respecto al HGB no es prácticamente convincente: degrada el ranking, más que triplica las falsas alarmas y no supera al prior constante en Brier. Un siguiente experimento debería definir presupuesto y coste de alertas, medir la estabilidad por ruta y validar calibración y detección en un período nuevo y suficientemente maduro. Agosto no debe volver a usarse para seleccionar.

Sin cambios en CSV, frontend, API u otros módulos; sin commits ni push.
