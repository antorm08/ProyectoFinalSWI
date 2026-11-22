from fastapi import APIRouter, Query

from app.schemas import DemandOutput
from app.services import demand_service

router = APIRouter(prefix="/demand", tags=["2. Series temporales"])


@router.get("/forecast", response_model=DemandOutput)
def forecast(horizonte: int = Query(30, ge=1, le=180), historico: int = Query(90, ge=7, le=600)):
    return demand_service.forecast(horizonte, historico)
