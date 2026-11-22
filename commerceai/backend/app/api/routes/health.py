from fastapi import APIRouter

from app.core import config
from app.db import olist

router = APIRouter(tags=["estado"])


@router.get("/health")
def health():
    """Estado de la API, de los datos y de cada modelo entrenado."""
    return {
        "status": "ok",
        "datos_olist": {name: olist.available(name) for name in olist.FILES},
        "modelos": {
            "retraso": config.DELAY_MODEL.exists(),
            "demanda": config.DEMAND_MODEL.exists(),
            "productos": config.PRODUCT_MODEL.exists(),
            "vendedores": config.SELLER_MODEL.exists(),
            "resumen": True,  # TextRank no requiere entrenamiento
        },
    }
