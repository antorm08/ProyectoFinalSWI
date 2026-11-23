# Manual de instalación – CommerceAI

> Plantilla. Ampliar con capturas de pantalla.

## 1. Requisitos
- Python 3.11 o 3.12
- Node.js 20 o superior
- (Opcional) Docker Desktop

## 2. Obtener los datos
Ver `data/README.md`.

## 3. Instalación del backend
## 4. Entrenamiento de los modelos

### Pronóstico de demanda (perfil semanal · Olist)

Necesita los CSV de Olist en `data/olist/` (§2). No hace falta PyTorch. Desde la raíz del proyecto:

```powershell
# 1) Auditar la serie sin entrenar: ventana, huecos de cobertura y cola truncada
backend\.venv\Scripts\python.exe -m ml.train_demand --check-data

# 2) Comparar candidatos y congelar el modelo (~15 min; escribe gráficas y métricas)
backend\.venv\Scripts\python.exe -m ml.evaluate_demand

# 3) Entrenar el artefacto que sirve la API (~1 min)
backend\.venv\Scripts\python.exe -m ml.train_demand

# 4) Opcional: banco de 18 algoritmos (~90 s), el que motivó el modelo actual
backend\.venv\Scripts\python.exe -m ml.benchmark_demand
```

- `evaluate_demand.py` escribe `docs/resultados_demanda/metricas.json` y cinco gráficas. **No sobrescribe**: si el
  directorio ya tiene resultados, falla a propósito. Para repetir, usa `--output` con otra carpeta o borra la anterior.
  `--rapido` recorta la rejilla y los orígenes; sirve para comprobar la tubería, no para publicar cifras.
- `train_demand.py` lee la especificación congelada de ese informe, mide el holdout reservado, reajusta con toda la
  ventana y guarda `backend/app/models/demand_model.joblib` más `docs/demanda_metricas.json`. Antes de activarlo
  comprueba que el artefacto recargado reproduce su propio pronóstico; si no, se detiene.
- Reinicia el backend después de entrenar. Comprobación: `http://localhost:8000/api/health` debe mostrar
  `"demanda": true`, y `GET /api/demand/forecast?horizonte=14` debe devolver `modelo_entrenado: true` con
  `inferior` y `superior` en cada punto futuro.

Pruebas del componente (desde la raíz):

```powershell
backend\.venv\Scripts\python.exe -m unittest ml.tests.test_demand -v
backend\.venv\Scripts\python.exe -m unittest ml.tests.test_demand_api -v
backend\.venv\Scripts\python.exe -m unittest ml.tests.test_demand_benchmark -v
```

### Clasificación de productos (CNN · Fashion-MNIST)

No necesita descargar datos a mano: Fashion-MNIST (~30 MB) se descarga sola en `data/fashion_mnist/` la primera vez.
Desde la raíz del proyecto, en PowerShell (sin activar el entorno; se usa su Python directamente):

```powershell
cd ml
..\backend\.venv\Scripts\python.exe train_products.py      # ~15 min en CPU
..\backend\.venv\Scripts\python.exe evaluate_products.py   # ~1 min, opcional: gráficos y robustez
```

- `train_products.py` genera `backend/app/models/product_cnn.pt` (no se sube a git) y
  `docs/clasificacion_productos_metricas.json`. Al final muestra el accuracy en test y si alcanza el objetivo (≥ 91 %).
  Opciones: `--epochs`, `--patience`, `--seed`, `--subset N` (prueba rápida con N imágenes).
- `evaluate_products.py` necesita `matplotlib` (`pip install -r ml/requirements.txt`) y escribe
  `docs/resultados_clasificacion_productos/`.
- Reinicia el backend después de entrenar para que cargue el modelo nuevo. Con Docker no hace falta reconstruir la
  imagen: la carpeta `backend/app/models` está montada en el contenedor.

Comprobación: `http://localhost:8000/api/health` debe mostrar `"productos": true`.

Pruebas del componente (desde la raíz):

```powershell
backend\.venv\Scripts\python.exe -m unittest ml.tests.test_products -v
```
## 5. Instalación del frontend
## 6. Instalación con Docker
## 7. Solución de problemas frecuentes

- **La activación del entorno falla con "la ejecución de scripts está deshabilitada"** (PowerShell): ejecuta una vez
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, o usa `backend\.venv\Scripts\python.exe` directamente.
- **Clasificación visual muestra "Clasificación no disponible"**: falta `backend/app/models/product_cnn.pt`. Entrena el
  modelo (§4) y reinicia el backend.
- **Aviso `FutureWarning: torch.jit... is deprecated`** al entrenar o al arrancar el backend: es informativo; el modelo
  TorchScript funciona con la versión de PyTorch instalada.
- **`ModuleNotFoundError: No module named 'ml'` al ejecutar las pruebas**: ejecútalas desde la raíz del proyecto con
  `python -m unittest ml.tests.<módulo>`.
- **El pronóstico de demanda dice "Sin modelo entrenado"** aunque el `.joblib` exista: el artefacto no es servible
  (formato antiguo, fichero corrupto o statsmodels incompatible). La API no falla, cae a repetir la última semana y deja
  el motivo en el registro del backend. Vuelve a ejecutar `python -m ml.train_demand`.
- **`FileExistsError` al lanzar `ml.evaluate_demand`**: es deliberado, para no pisar resultados ya publicados. Usa
  `--output` con otra carpeta.
- **Las pruebas E2E no arrancan el navegador en Linux**: `playwright.config.ts` usa Edge por omisión. Ejecuta
  `PW_CHANNEL=chromium npm run test:e2e` después de `npx playwright install chromium`.
