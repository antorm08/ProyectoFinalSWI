from fastapi import APIRouter, HTTPException, Query

from app.schemas import ReviewInsights, ReviewOptions, SummaryInput, SummaryOutput
from app.services import summary_service

router = APIRouter(prefix="/reviews", tags=["5. Resumen de texto"])

N_FRASES = Query(5, ge=1, le=15, description="Frases representativas por sección (reclamos y elogios)")


@router.post("/summarize", response_model=SummaryOutput)
def summarize(data: SummaryInput):
    """Resume un texto pegado (reseñas en español o portugués) y detecta sus temas."""
    return summary_service.summarize(data.texto, data.n_oraciones)


@router.get("/options", response_model=ReviewOptions)
def options(limite: int = Query(30, ge=1, le=200)):
    """Vendedores y categorías con más reseñas comentadas (para los selectores del frontend)."""
    return summary_service.review_options(limite)


@router.get("/overview", response_model=ReviewInsights)
def overview(n_oraciones: int = N_FRASES):
    """Reclamos y elogios más representativos de toda la plataforma."""
    result = summary_service.summarize_platform(n_oraciones)
    if result is None:
        raise HTTPException(404, "Datos de reseñas de Olist no cargados (ver data/README.md)")
    return result


@router.get("/seller/{seller_id}", response_model=ReviewInsights)
def summarize_seller(seller_id: str, n_oraciones: int = N_FRASES):
    result = summary_service.summarize_seller(seller_id, n_oraciones)
    if result is None:
        raise HTTPException(404, "Sin reseñas para ese vendedor (o datos de Olist no cargados)")
    return result


@router.get("/category/{categoria}", response_model=ReviewInsights)
def summarize_category(categoria: str, n_oraciones: int = N_FRASES):
    result = summary_service.summarize_category(categoria, n_oraciones)
    if result is None:
        raise HTTPException(404, "Sin reseñas para esa categoría (o datos de Olist no cargados)")
    return result
