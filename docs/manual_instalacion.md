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
