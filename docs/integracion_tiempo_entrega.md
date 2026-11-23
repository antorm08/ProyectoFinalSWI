# Integración experimental de la estimación de entrega

El Módulo 1 muestra una estimación de duración en la ruta existente `/retrasos`. Usa el **HistGradientBoostingRegressor sin plazo prometido** seleccionado en [la evaluación de Olist](evaluacion_tiempo_entrega.md). No reemplaza el clasificador histórico ni interviene en los otros módulos. No es una función lista para producción.

## Qué exige realmente el modelo

Se inspeccionó el artefacto ajustado, no solo el esquema del conjunto de entrenamiento. `Pipeline.feature_names_in_` conserva los diez nombres del DataFrame original (incluido `dias_estimados`), pero el `ColumnTransformer` del candidato selecciona únicamente:

| Entrada del modelo | Origen en la API |
|---|---|
| `precio`, `flete`, `n_items` | Totales del pedido |
| `peso_g` | Peso total en gramos; `0` se transforma a NaN, como en entrenamiento |
| `estado_cliente`, `estado_vendedor` | Siglas brasileñas normalizadas; el origen es representativo si hay varios |
| `mismo_estado` | Igualdad entre las dos siglas, calculada en el backend |
| `mes`, `dia_semana` | Derivados de la fecha de compra o indicados explícitamente; lunes = 0 |

La predicción de control dio el mismo resultado con `dias_estimados` omitido y con valor 999. Por ello el contrato anterior **no se reutiliza como requisito de ETA**: `DelayInput` todavía exige plazo para clasificación, pero el formulario y `DeliveryDurationInput` no lo aceptan ni lo envían. Las entradas no incluyen fechas reales de entrega, reseñas ni estado final.

## Contratos y comportamiento

- `POST /api/delay/predict-duration`: requiere importes y peso finitos no negativos, cantidad entera positiva, siglas brasileñas y **o bien** `fecha_compra` (`YYYY-MM-DD`, fecha local sin hora) **o bien** `mes` y `dia_semana`. Un contrato inválido devuelve `422`.
- Cuando hay fecha, la API muestra la fecha de compra más la parte entera de la duración predicha. Es una fecha aproximada, sin estimar hora: una compra real a última hora podría desplazarla un día. La duración conserva decimales. Sin fecha devuelve `fecha_llegada_estimada: null`; no se usa la fecha de consulta ni se fabrica una referencia.
- Devuelve `503` cuando el artefacto falta, no coincide con el binario aprobado o no se puede ejecutar. No hay heurística ni valor alternativo. La UI muestra el mensaje de modelo no disponible y ningún resultado.
- `POST /api/delay/predict` mantiene el contrato y el servicio anteriores (`probabilidad_retraso`, `nivel_riesgo`, `modelo_entrenado`). El clasificador no se invoca al estimar duración. `/api/health` conserva `modelos.retraso` y agrega `modelos.tiempo_entrega` como **indicador de presencia de archivo**, no de integridad, precisión o disponibilidad de inferencia.
- La interfaz mantiene sus seis rutas. El menú y el tablero identifican la nueva función como **Tiempo de entrega**; las rutas y vistas de demanda, imágenes, vendedores y reseñas no se modifican.

Ejemplo de solicitud:

```json
{
  "precio": 120,
  "flete": 20,
  "peso_g": 1500,
  "n_items": 1,
  "estado_cliente": "RJ",
  "estado_vendedor": "SP",
  "fecha_compra": "2018-08-06"
}
```

## Carga y procedencia

El backend usa `delivery_regression_experimental.joblib` en el directorio de modelos configurado por el servidor, no una ruta o un archivo suministrado por el cliente. Antes de deserializar verifica tamaño (**326.117 bytes**) y SHA-256 (`4a698a3771aa50c0e945d24db434fa7b074b3a16823201128966ba54d804df20`); lee el binario una vez por carga y pasa esos mismos bytes a Joblib. Después valida el estimador y la selección de columnas. El resultado de una carga correcta se guarda en caché por ruta y metadatos del archivo. La comprobación de hash solo autoriza este artefacto local ya revisado: Joblib no debe usarse con archivos sin procedencia confiable.

El entrenamiento utilizó scikit-learn 1.9.1. No se admite deserializar si scikit-learn informa incompatibilidad de versión. El binario está ignorado por Git: entregar el original por separado para reproducir la demo. Una regeneración que produzca bytes distintos requiere revisar compatibilidad, métricas y hash antes de autorizarlo; no se debe reemplazar la validación con un hash de procedencia no verificada.

### Cómo leer los resultados

Los MAE mostrados son **4,60 días en desarrollo** y **3,31 días en agosto de 2018 retrospectivo**. Son errores medios en cohortes, no un intervalo de confianza individual ni una garantía. La regresión subestimó las entregas largas y carece de una validación externa nueva. Estima duración; no promete fechas, no estima probabilidad de retraso ni mejora un nivel de servicio comercial demostrado.

## Ejecución de las pruebas

Desde la raíz:

```powershell
.\backend\.venv\Scripts\python.exe -B -m unittest discover -s ml/tests
```

Para las pruebas de navegador, iniciar la API desde `backend/`, y luego compilar y ejecutar Playwright desde `frontend/`:

```powershell
# Terminal 1, desde backend/
.\.venv\Scripts\python.exe -B -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# Terminal 2, desde frontend/
npm.cmd run build
npm.cmd run test:e2e
```

Las pruebas de ETA con y sin fecha usan el backend y el artefacto reales. El caso de error en Playwright simula un `503` para comprobar la presentación; las pruebas ASGI provocan ausencia o hash incorrecto de verdad. Se verifican también la validación, la ausencia del plazo, la previsión obsoleta al editar el pedido y el contrato del clasificador.

## Evidencia desde la interfaz

Las capturas provienen de Playwright contra la API local, sin sustituir la respuesta exitosa:

| Escenario | Resultado observado | Evidencia |
|---|---|---|
| Compra fechada el 6 de agosto de 2018; SP → RJ, BRL 120 + 20 de flete, 1.500 g y un artículo | HTTP 200: `10.90887913460709` días; llegada `2018-08-16`. La vista muestra **10,91 días**. | [Escritorio, con fecha](evidencia_tiempo_entrega/prediccion-con-fecha-escritorio.png) |
| Mismo pedido, sin fecha exacta; agosto y lunes | HTTP 200: misma duración; llegada `null`. No se muestra una fecha de llegada. | [Móvil, sin fecha](evidencia_tiempo_entrega/prediccion-sin-fecha-movil.png) |

**Verificación:** 70 pruebas de Python y 72 de Playwright aprobadas (escritorio, escritorio compacto, tableta y móvil), incluidas comprobaciones de accesibilidad y las seis vistas existentes. La compilación de frontend pasó con un aviso de tamaño del paquete. Los SHA-256 de los CSV y de los artefactos existentes no cambiaron. No hubo reentrenamiento, commits ni push.
