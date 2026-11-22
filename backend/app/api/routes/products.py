from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import ProductOutput
from app.services import product_service

router = APIRouter(prefix="/products", tags=["3. Clasificación de imágenes"])


@router.post("/classify", response_model=ProductOutput)
async def classify(imagen: UploadFile = File(...)):
    if not (imagen.content_type or "").startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")
    return product_service.classify(await imagen.read())
