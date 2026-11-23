# Preparación metodológica de Olist

**Fecha:** 22 de noviembre de 2026.  
**Alcance:** retrasos y segmentación de vendedores, sobre la copia principal.  
**Estado:** preparación verificada y lista para un **entrenamiento exploratorio controlado**. No se ha entrenado ningún modelo ni se certifica precisión o preparación para producción.

Este documento complementa `docs/validacion_olist.md`, que conserva las estadísticas del código anterior a las correcciones.

## 1. Archivos modificados

| Archivo | Cambio |
|---|---|
| `ml/common.py` | Objetivo calendario compartido, duraciones sin truncar, auditoría y joins con cardinalidad comprobada |
| `ml/train_delay.py` | Pesos desconocidos, origen representativo determinista, días calendario, selección explícita de variables, evaluación temporal y modo `--check-data` |
| `ml/temporal.py` | Particiones cronológicas con madurez de etiquetas, intervalos de espera y registro de todas las exclusiones |
| `ml/train_sellers.py` | Unidad vendedor-pedido, agregación previa de reseñas y modo `--check-data` |
| `backend/app/services/delay_service.py` | Peso cero tratado como desconocido antes de inferir, igual que en preparación |
| `ml/tests/test_preparation.py` | 31 pruebas controladas |
| `ml/tests/test_real_olist.py` | 9 pruebas con los CSV reales e integridad de artefactos |
| `docs/validacion_olist.md` | Referencia al estado corregido, sin reescribir el diagnóstico original |
| `docs/preparacion_olist.md` | Decisiones, resultados, comandos y limitaciones |

Sin cambios en frontend, endpoints, esquemas HTTP, CSV, modelos existentes, series temporales o la carpeta duplicada `commerceai/`. No se hicieron commits ni push.

## 2. Objetivo de retraso único y compartido

```python
retrasado = (
    order_delivered_customer_date.dt.normalize()
    > order_estimated_delivery_date.dt.normalize()
).astype(int)
```

Una entrega a las 23:59:59 del día prometido es puntual. Una entrega en el día siguiente es tardía. Se usa la fecha local representada en el CSV, sin inventar zonas horarias.

La implementación reside en `delivered_orders()` y se utiliza tanto en retrasos como en vendedores. Los servicios consumen los artefactos resultantes; no contienen otra comparación de fechas que deba sustituirse.

| Indicador | Antes | Después |
|---|---:|---:|
| Pedidos originales | 99.441 | 99.441 |
| Estado delivered | 96.478 | 96.478 |
| Entregados sin fechas de objetivo | 8 | 8 |
| Elegibles con objetivo | 96.470 | 96.470 |
| Retrasados | 7.826 | **6.534** |
| Puntuales | 88.644 | **89.936** |
| Tasa de retraso | 8,11 % | **6,77 %** |

Cambian exactamente **1.292 etiquetas**, correspondientes a entregas del mismo día prometido. Los ocho casos sin entrega real no se imputan como puntuales; se contabilizan como no etiquetables.

Las fechas imposibles de compra/entrega o día prometido anterior a compra provocan un error explícito, no un borrado silencioso. Las anomalías auxiliares de aprobación/transportista del informe original no se utilizan como predictores ni se eliminan automáticamente; la etiqueta sigue siendo calculable para esos pedidos.

## 3. Características de retrasos

Se mantienen las ocho variables numéricas y dos categóricas existentes:

```text
precio, flete, peso_g, n_items, dias_estimados,
mes, dia_semana, mismo_estado,
estado_cliente, estado_vendedor
```

`order_id`, comprador, fechas, disponibilidad de etiqueta y cantidad de estados se conservan solo como metadatos de auditoría/partición. El `ColumnTransformer` selecciona expresamente las diez variables y descarta el resto, incluso si por error recibe el DataFrame completo.

### 3.1 Pesos faltantes y cero

- Peso no informado o exactamente cero se considera **desconocido**, no un objeto sin masa.
- Se suma una vez el peso de cada fila de artículo. Varios artículos del mismo producto cuentan varias veces, porque son unidades distintas.
- Si falta el peso de **cualquier artículo**, el peso total queda `NaN`; no se presenta una suma parcial como total conocido.
- Se conservan los pedidos. HistGradientBoosting admite `NaN` nativamente: **no se imputa con la mediana global, ni se usa validación/prueba para completar valores**.
- Valores negativos, infinitos, importes faltantes o referencias rotas producen errores de validación. No se inventan importes.
- En inferencia, el contrato ya permite `peso_g=0`; ahora se interpreta como desconocido y se transforma internamente a `NaN`. El payload y la respuesta no cambian. La heurística provisional no utiliza peso, por lo que conserva su resultado.

| Resultado sobre los CSV reales | Conteo |
|---|---:|
| Pedidos sin peso conocido por faltantes en catálogo | 16 |
| Pedidos adicionales con peso explícitamente cero | 6 |
| Pesos agregados cero, antes | 22 |
| Pesos desconocidos `NaN`, después | **22** |
| Pesos agregados cero, después | **0** |
| Pedidos descartados por esta política | **0** |

Los 22 faltantes se distribuyen en 16 de entrenamiento, 0 de validación, 4 de prueba y 2 de los intervalos excluidos.

### 3.2 Pedidos multivendedor

El contrato solo puede representar un `estado_vendedor`. Para no perder observaciones ni introducir un código artificial:

1. Sumar el valor de artículos por **pedido y estado de origen**, comparando importes redondeados a centavos.
2. Seleccionar el estado con mayor valor de artículos.
3. Si hay empate, seleccionar el de mayor cantidad de artículos.
4. Si continúa el empate, usar la sigla de estado en orden alfabético.

El estado resultante es el **origen representativo por valor**, no una afirmación de que todos los artículos salieron de allí. Precio, flete, peso y cantidad siguen considerando **todos los artículos y vendedores**.

En los 96.470 pedidos elegibles hay **509 con varios estados de origen**. La política nueva cambia el origen de 259 pedidos frente a tomar la primera fila; conserva los 509. Cambiar el orden de las filas no altera el origen elegido, como comprueban las pruebas.

**Límite del esquema:** no puede expresar rutas múltiples ni atribuir un retraso a un envío concreto. La solución aplicada es una aproximación compatible. Una representación completa necesitaría una lista de orígenes/envíos o una evaluación por envío, con cambios de contrato que **no se han implementado**. En integraciones manuales se debe suministrar el estado representativo según la misma política.

### 3.3 Días prometidos y tiempo real

- `dias_estimados = día_prometido - día_de_compra`, en días calendario enteros.
- Antes se restaba un timestamp de compra a una promesa a medianoche y se truncaba con `.dt.days`, perdiendo casi siempre un día.
- Cambian **96.469 pedidos**; el rango pasa de 2–155 a **3–156 días**. El pedido restante se compró exactamente a medianoche.
- El formulario conserva el mismo campo entero; debe introducirse la diferencia entre fechas calendario, no la duración truncada desde la hora de compra.
- `dias_entrega` para vendedores usa horas transcurridas divididas por 24, sin truncar: puede ser decimal. El esquema existente ya admite ese valor.

### 3.4 Joins y orden de filas

- Se comprueba la unicidad de pedido, producto, vendedor, cliente e ítem compuesto según su relación.
- Las referencias obligatorias se unen con `how="left"` y `validate="many_to_one"`, comprobando coincidencias y conteos.
- Una referencia ausente o repetida provoca un error. No se usan inner joins que descarten pedidos ni joins que multipliquen ingresos.
- Resultado: **96.470 pedidos antes y después de las uniones**, con 110.189 artículos elegibles conservados en sus agregaciones.

## 4. Evaluación cronológica con etiquetas disponibles

### 4.1 Cortes predeterminados

Los intervalos de compra son cerrados por la izquierda y abiertos por la derecha. Todos los límites son configurables mediante CLI.

| Uso | Compra | Etiqueta disponible antes de |
|---|---|---|
| Entrenamiento | Antes de 2018-05-01 | 2018-06-01 |
| Espera de maduración | Mayo de 2018 | No se incorpora al fit |
| Validación | 2018-06-01 a 2018-07-01, exclusivo | 2018-08-01 |
| Espera de maduración | Julio de 2018 | No se incorpora al fit |
| Prueba final | 2018-08-01 a 2018-09-01, exclusivo | 2018-10-18 |

Se deja aproximadamente un mes para observar entregas, en lugar de evaluar una cohorte inmediatamente después de sus compras. La cola de septiembre/octubre, de escasa cobertura, no se usa como ventana de compras de prueba.

`label_available_at` se fija al timestamp de entrega real. El CSV no contiene la fecha de ingestión ni un historial de cambios de estado: es un **proxy conservador de evento disponible**, no una prueba de la disponibilidad efectiva en un sistema de producción. Si existieran retrasos de ingestión, habría que incorporar sus timestamps.

### 4.2 Conteos medidos

| Conjunto | Pedidos | Retrasados | Tasa |
|---|---:|---:|---:|
| Entrenamiento | **71.023** | 5.331 | 7,51 % |
| Validación | **6.070** | 51 | 0,84 % |
| Prueba | **6.351** | 393 | 6,19 % |

Últimas etiquetas utilizadas:

- Entrenamiento: 2018-05-31 00:09:39, anterior a validación.
- Validación: 2018-07-31 20:36:42, anterior a prueba.
- Prueba: 2018-09-25 00:47:25, anterior al cierre observacional.

Las observaciones no utilizadas se conservan en `TemporalSplit.excluded`, con motivo, en memoria:

| Motivo | Pedidos | Retrasados |
|---|---:|---:|
| Espera de maduración de entrenamiento | 6.749 | 443 |
| Espera de maduración de validación | 6.156 | 208 |
| Etiqueta de entrenamiento aún no disponible al corte | 95 | 88 |
| Etiqueta de validación aún no disponible al corte | 26 | 20 |

**71.023 + 6.070 + 6.351 + 6.749 + 6.156 + 95 + 26 = 96.470.** No hay pedidos compartidos entre conjuntos ni filas perdidas. No se escriben exclusiones sobre los CSV originales.

**Limitación de censura:** un mes no garantiza observar entregas extremadamente largas. Junio tiene 6.096 elegibles y 71 retrasos en el histórico completo, pero 26 entregas —20 tardías— aún no se conocían al iniciar la prueba. No se usan sus etiquetas futuras para seleccionar un modelo. La tasa de validación observada no representa sin sesgo la tasa completa de junio. Esta limitación, y el cambio de prevalencia entre meses, deben acompañar las métricas; ampliar los tiempos de espera o implementar evaluación con censura requerirá una decisión posterior.

### 4.3 Qué se ajustará al entrenar posteriormente

- Se conserva **HistGradientBoostingClassifier**, `class_weight="balanced"`, `max_iter=300` y semilla 42.
- `early_stopping=False` evita la validación interna aleatoria del estimador. No se sustituye el algoritmo.
- Solo se ajustan el codificador categórico y el clasificador con **entrenamiento**.
- No se balancean ni mezclan aleatoriamente validación y prueba; sus prevalencias se conservan.
- Se mantiene el umbral 0,5 predefinido, sin optimizarlo sobre prueba.
- El script queda preparado para informar ROC-AUC, average precision, Brier y reporte de clasificación por conjunto. **No se han calculado estas métricas**, porque no se ejecutó ningún entrenamiento.
- No hay reajuste automático posterior con validación/prueba. El artefacto futuro corresponderá al modelo evaluado, ajustado solo con entrenamiento.
- Los compradores/vendedores conocidos pueden aparecer en distintos períodos. No se afirma generalización a compradores nuevos; las identidades no se usan como variables.
- Las probabilidades ponderadas por clases no deben considerarse calibradas sin una evaluación posterior específica.

## 5. Vendedores: unidad vendedor-pedido

Proceso:

1. Identificar los mismos 96.470 pedidos entregados con objetivo calculable.
2. Filtrar sus artículos: **110.189 filas**.
3. Agrupar artículos por `seller_id, order_id`, sumando `price`: **97.811 parejas**.
4. Promediar primero las múltiples notas de cada pedido: una fila de nota por `order_id`.
5. Unir esa nota a cada pareja con cardinalidad N:1: siguen siendo **97.811 filas**.
6. Calcular por vendedor las medias de notas, retraso y duración: cada pedido del vendedor tiene el mismo peso.
7. Conservar `facturacion` como suma de importes de artículos y `n_pedidos` como cantidad de parejas.

Se permiten `review_id` compartidos entre pedidos; se exige unicidad de `review_id, order_id`. No se une cada artículo con cada reseña.

No se inventa satisfacción: si faltan todas las notas de un vendedor, se registra como no elegible para el vector completo. Si solo faltan algunas, se promedian las disponibles. Hay **660 parejas sin nota**; antes eran 827 filas de artículo sin nota, que no son la misma unidad.

| Indicador | Antes | Después |
|---|---:|---:|
| Vendedores con pedidos elegibles | 2.970 | 2.970 |
| Menos de cinco pedidos | 1.204 | 1.204 |
| Elegibles para K-Means | **1.766** | **1.766** |
| Excluidos por métricas incompletas tras mínimo | 0 | 0 |
| Unidad de las medias | Artículo | Vendedor-pedido |

Comparación completa con el script anterior, incluyendo la nueva regla de retraso y duración decimal:

| Característica | Vendedores con cambio > 1e-9 | Diferencia absoluta máxima |
|---|---:|---:|
| `n_pedidos` | 0 | 0 |
| `facturacion` | 0 | Menos de 1e-9 BRL por suma en punto flotante |
| `nota_media` | 1.138 | 1,3583 puntos |
| `tasa_retraso` | 985 | 0,50, equivalente a 50 puntos porcentuales |
| `dias_entrega_medio` | 1.766 | 7,3865 días |

Las diferencias de retraso y duración no deben atribuirse solo a ponderación: también cambia el objetivo a calendario y se conservan horas. La suma comercial y el número de pedidos no cambian.

Se conserva K-Means con StandardScaler. No se ha ajustado ninguno. Siguen pendientes justificar `k=4`, evaluar estabilidad y revisar los nombres asignados a los grupos; estas decisiones no se inventan durante la preparación.

## 6. Compatibilidad y límites que no se ocultan

- Los contratos HTTP y las columnas esperadas por los servicios de inferencia se mantienen.
- `peso_g=0` significa desconocido internamente; no se envía `NaN` en JSON ni se cambia el esquema.
- El esquema de retrasos sigue representando un solo estado de origen. Los 509 casos multiorigen usan el representante documentado, no un origen completo ficticio.
- La satisfacción de un pedido multivendedor se atribuye a cada vendedor participante por falta de reseñas específicas de cada envío. No equivale a responsabilidad individual demostrada.
- La facturación es suma de valores de artículos de pedidos elegibles, no cobros netos confirmados.
- El clustering es retrospectivo: usa entregas y reseñas históricas. No se deben reutilizar estos segmentos como predictores de compras pasadas sin calcularlos solo con historia anterior.
- La población de retrasos continúa condicionada a pedidos finalmente entregados; no es un predictor de cancelación.
- Los atributos de catálogo son una instantánea sin historial de versiones; no puede acreditarse desde estos CSV que nunca cambiaron después de comprar.
- No se sobrescribieron artefactos anteriores. Los modelos de retrasos/vendedores siguen pendientes; si se emplearan artefactos antiguos en otra instalación, deberán regenerarse antes de atribuirles estas reglas.
- Para conservar la selección original, las 1.373 anomalías auxiliares de cronología no se eliminan automáticamente. Debe evaluarse sensibilidad a esos casos cuando se autorice entrenar.

## 7. Pruebas y comandos ejecutados

Desde la raíz, con el entorno Python local:

```powershell
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s ml/tests -v
.\backend\.venv\Scripts\python.exe -B ml/train_delay.py --check-data
.\backend\.venv\Scripts\python.exe -B ml/train_sellers.py --check-data
```

**Resultado: 40 pruebas aprobadas, 0 errores y 0 omitidas**, incluyendo las 9 de los CSV reales.

Cobertura:

- Entregas durante el día prometido y en días posteriores.
- Días calendario y duración con horas.
- Múltiples artículos y vendedores; empate por importe, cantidad y sigla.
- Pesos faltantes, cero y sumas parciales.
- Varias reseñas por pedido y `review_id` compartido entre pedidos.
- Joins con referencias repetidas/huérfanas; conservación de artículos, pedidos e importes.
- Separación cronológica, límites exactos, etiquetas tardías, intervalos de maduración y conservación de todas las exclusiones.
- Preprocesamiento ajustado solo con el conjunto controlado de entrenamiento, sin metadatos ni categorías futuras.
- Compatibilidad del servicio al recibir peso cero, usando un predictor simulado **solo en la prueba**, no un modelo entrenado ni resultados persistidos.
- Hashes de los seis CSV iguales a los del informe de validación.
- Manifestación de artefactos sin cambios antes y después de la batería.
- Ejecución de los dos `main(check_only=True)` con `Pipeline.fit`, `KMeans.fit`, `HistGradientBoostingClassifier.fit`, `joblib.dump` y `DataFrame.to_csv` bloqueados por pruebas: no se intenta entrenar ni escribir modelos/tablas.

El único `fit` ejecutado en pruebas es el del **codificador de categorías** sobre unos pocos casos controlados para comprobar aislamiento del preprocesamiento. No se ajustó ningún estimador predictivo, K-Means ni escalador de clustering.

**Advertencia operativa:** ejecutar los scripts de entrenamiento **sin `--check-data` sí entrenaría modelos**. No se ejecutaron de esa forma.

## 8. Dictamen

Los problemas confirmados de objetivo, pesos, agregación y partición ya están corregidos con políticas explícitas, reproducibles y comprobadas. Los datos preparados están **listos para autorizar entrenamiento y evaluación exploratorios**, sin necesidad de cambiar los contratos.

Esto no significa que los modelos ya existan ni que su precisión esté validada. La censura residual, la representatividad del origen único, la prevalencia baja en validación y la calidad de etiquetas requieren interpretar con cautela las métricas futuras antes de producción.
