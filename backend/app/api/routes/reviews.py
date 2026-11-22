from fastapi import APIRouter, HTTPException, Query

from app.schemas import SummaryInput, SummaryOutput
from app.services import summary_service

router = APIRouter(prefix="/reviews", tags=["5. Resumen de texto"])


@router.post("/summarize", response_model=SummaryOutput)
def summarize(data: SummaryInput):
    return summary_service.summarize(data.texto, data.n_oraciones)


@router.get("/seller/{seller_id}", response_model=SummaryOutput)
def summarize_seller(seller_id: str, n_oraciones: int = Query(5, ge=1, le=15)):
    result = summary_service.summarize_seller(seller_id, n_oraciones)
    if result is None:
        raise HTTPException(404, "Sin reseñas para ese vendedor (o datos de Olist no cargados)")
    return result
