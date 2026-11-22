"""Configuración central: rutas a datos y modelos entrenados."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]          # raíz del repo: commerceai/
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
OLIST_DIR = DATA_DIR / "olist"
MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).resolve().parents[1] / "models"))

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

# Nombres de los artefactos generados por los scripts de ml/
DELAY_MODEL = MODELS_DIR / "delay_model.joblib"
DEMAND_MODEL = MODELS_DIR / "demand_model.joblib"
PRODUCT_MODEL = MODELS_DIR / "product_cnn.pt"
SELLER_MODEL = MODELS_DIR / "seller_kmeans.joblib"
SELLER_TABLE = MODELS_DIR / "seller_clusters.csv"
