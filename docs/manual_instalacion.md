# Manual de instalación – CommerceAI

## 1. Requisitos
- Python 3.11 o 3.12
- Node.js 20 o superior
- (Opcional) Docker Desktop

## 2. Obtener los datos
Ver `data/README.md`.

## 3. Instalación del backend

Desde la raíz del proyecto, en PowerShell:

```powershell
python -m venv backend/.venv
backend\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
backend\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
backend\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --port 8000
```

Comprueba el arranque en <http://localhost:8000/docs>. Usa los comandos de entrenamiento en otra terminal,
desde la raíz. Instala únicamente artefactos de confianza y reinicia la API tras cambiarlos.

## 4. Entrenamiento de los modelos

### Pronóstico de demanda (SARIMAX · Olist)

Necesita los CSV de Olist en `data/olist/` (§2). No hace falta PyTorch. Desde la raíz del proyecto:

```powershell
# 1) Auditar la serie sin entrenar: ventana, huecos de cobertura y cola truncada
backend\.venv\Scripts\python.exe -m ml.train_demand --check-data

# 2) Entrenar desde la especificación congelada, sin sobrescribir la versión instalada
backend\.venv\Scripts\python.exe -m ml.train_demand --output candidatos/demand_model.joblib --metrics candidatos/demanda.json
```

- `evaluate_demand.py` escribe `docs/resultados_demanda/metricas.json` y cinco gráficas. **No sobrescribe**: si el
  directorio ya tiene resultados, falla a propósito. Para repetir, usa `--output` con otra carpeta o borra la anterior.
  `--rapido` recorta la rejilla y los orígenes; sirve para comprobar la tubería, no para publicar cifras.
- `train_demand.py` lee la especificación congelada de ese informe, mide el holdout reservado, reajusta con toda la
  ventana y guarda el bundle y el informe en las rutas elegidas. Antes de activarlo
  comprueba que el artefacto recargado reproduce su propio pronóstico; si no, se detiene.
- Reinicia el backend después de entrenar. Comprobación: `http://localhost:8000/api/health` debe mostrar
  `"demanda": true`, y `GET /api/demand/forecast?horizonte=14` debe devolver `modelo_entrenado: true` con
  `inferior` y `superior` en cada punto futuro.

Pruebas del componente (desde la raíz):

```powershell
backend\.venv\Scripts\python.exe -m unittest ml.tests.test_demand -v
backend\.venv\Scripts\python.exe -m unittest ml.tests.test_demand_api -v
```

### Clasificación de productos (CNN · Fashion-MNIST)

No necesita descargar datos a mano: Fashion-MNIST (~30 MB) se descarga sola en `data/fashion_mnist/` la primera vez.
Desde la raíz del proyecto, en PowerShell (sin activar el entorno; se usa su Python directamente):

```powershell
backend\.venv\Scripts\python.exe -m ml.train_products --threads 4 --output candidatos/product_cnn.pt --metrics candidatos/productos.json
```

- `train_products.py` genera TorchScript y un informe con su SHA-256. Al final muestra el accuracy en test
  y si alcanza el objetivo (≥ 91 %).
  Opciones: `--epochs`, `--patience`, `--seed`, `--subset N` (prueba rápida con N imágenes).
- `evaluate_products.py` necesita `matplotlib` (`pip install -r ml/requirements.txt`) y escribe
  `docs/resultados_clasificacion_productos/`.
- Reinicia el backend después de entrenar para que cargue el modelo nuevo. Con Docker no hace falta reconstruir la
  imagen: la carpeta `backend/app/models` está montada en el contenedor.

Comprobación: `http://localhost:8000/api/health` debe mostrar `"productos": true`. Envía una imagen a
`POST /api/products/classify`: comprueba `modelo_entrenado: true` y que `X-Model-SHA256` coincide con el informe.

Pruebas del componente (desde la raíz):

```powershell
backend\.venv\Scripts\python.exe -m unittest ml.tests.test_products -v
```

### Segmentación de vendedores (K-Means · Olist)

```powershell
backend\.venv\Scripts\python.exe -m ml.train_sellers --models-dir candidatos/vendedores --metrics candidatos/vendedores.json
```

Instala juntos `seller_kmeans.joblib` y `seller_clusters.csv` después de verificarlos. `GET /api/sellers/clusters`
debe devolver tres grupos que sumen 1.766 vendedores; `POST /api/sellers/assign` usa el modelo para asignar un perfil.

Conserva el binario aprobado de [ETA](integracion_tiempo_entrega.md). TextRank no requiere entrenamiento.
Consulta [cómo respaldar y activar los candidatos](auditoria_final_modulos.md#preparar-otra-instalación).

## 5. Instalación del frontend

En otra terminal, desde la raíz:

```powershell
cd frontend
npm ci
npm run dev
```

Abre <http://localhost:5173>. Para verificar la compilación, ejecuta `npm run build` en esa misma carpeta.

## 6. Instalación con Docker

Con los datos y artefactos ya instalados en el host, ejecuta `docker compose up --build` desde la raíz.
El frontend queda en <http://localhost:8080>; el backend monta los CSV y los artefactos en solo lectura.

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
  el motivo en el registro del backend. Genera un candidato con rutas nuevas como en §4.
- **`FileExistsError` al lanzar `ml.evaluate_demand`**: es deliberado, para no pisar resultados ya publicados. Usa
  `--output` con otra carpeta.
- **Las pruebas E2E no arrancan el navegador en Linux**: `playwright.config.ts` usa Edge por omisión. Ejecuta
  `PW_CHANNEL=chromium npm run test:e2e` después de `npx playwright install chromium`.
