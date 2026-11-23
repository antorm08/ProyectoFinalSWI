from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import PRODUCT_MAX_BYTES
from app.schemas import ProductOutput
from app.services import product_service

router = APIRouter(prefix="/products", tags=["3. Clasificación de imágenes"])


@router.post("/classify", response_model=ProductOutput,
             responses={400: {"description": "El archivo no es una imagen válida"},
                        413: {"description": "La imagen supera el tamaño máximo"},
                        422: {"description": "No se detecta ninguna prenda en la imagen"}})
async def classify(imagen: UploadFile = File(...)):
    if not (imagen.content_type or "").startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")
    data = await imagen.read(PRODUCT_MAX_BYTES + 1)
    if len(data) > PRODUCT_MAX_BYTES:
        raise HTTPException(413, f"La imagen supera el máximo de {PRODUCT_MAX_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(400, "El archivo está vacío")
    try:
        return product_service.classify(data)
    except product_service.InvalidImageError as exc:
        raise HTTPException(400, str(exc)) from None
    except product_service.NoObjectError as exc:
        raise HTTPException(422, str(exc)) from None
