from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from app.schemas import DelayInput, DelayOutput, DeliveryDurationInput, DeliveryDurationOutput
from app.services import delay_service, delivery_duration_service

router = APIRouter(prefix="/delay", tags=["1. Predicción de retraso"])


@router.post("/predict", response_model=DelayOutput)
def predict_delay(data: DelayInput):
    return delay_service.predict(data)


class DurationRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def handle(request: Request):
            try:
                return await handler(request)
            except RequestValidationError as error:
                # Do not echo invalid input (e.g. a number decoded as infinity).
                detail = [{key: entry[key] for key in ("loc", "msg", "type")} for entry in error.errors()]
                return JSONResponse(status_code=422, content={"detail": detail})

        return handle


duration_router = APIRouter(route_class=DurationRoute)


@duration_router.post(
    "/predict-duration",
    summary="Estima la duración de entrega (experimental)",
    response_model=DeliveryDurationOutput,
    responses={503: {"description": "Artefacto experimental no disponible o incompatible"}},
)
def predict_duration(data: DeliveryDurationInput):
    try:
        return delivery_duration_service.predict(data)
    except delivery_duration_service.ModelUnavailable:
        raise HTTPException(503, "Modelo de tiempo de entrega no disponible. No se generó una estimación.") from None
    except delivery_duration_service.ArrivalDateOutOfRange:
        raise HTTPException(422, "La fecha estimada queda fuera del rango de fechas admitido.") from None


router.include_router(duration_router)
