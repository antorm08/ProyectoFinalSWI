from fastapi import APIRouter

from app.schemas import DelayInput, DelayOutput
from app.services import delay_service

router = APIRouter(prefix="/delay", tags=["1. Predicción de retraso"])


@router.post("/predict", response_model=DelayOutput)
def predict_delay(data: DelayInput):
    return delay_service.predict(data)
