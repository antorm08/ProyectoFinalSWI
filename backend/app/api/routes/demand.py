from fastapi import APIRouter, Query

from app.schemas import DemandOutput
from app.services import demand_service

router = APIRouter(prefix="/demand", tags=["2. Series temporales"])

#: La serie tiene unas 84 semanas y ningún ciclo anual completo, así que no hay
#: estacionalidad anual que aprender: pronosticar más de 90 días sería extrapolar
#: sin evidencia. Ver docs/evaluacion_demanda.md, sección de limitaciones.
HORIZONTE_MAXIMO = 90


@router.get("/forecast", response_model=DemandOutput,
            summary="Pronóstico de la demanda diaria de pedidos")
def forecast(
    horizonte: int = Query(30, ge=1, le=HORIZONTE_MAXIMO,
                           description=f"Días a pronosticar (máximo {HORIZONTE_MAXIMO})"),
    historico: int = Query(90, ge=7, le=600, description="Días de histórico a devolver"),
    nivel: float = Query(0.8, ge=0.5, lt=1,
                         description="Nivel nominal de la banda; se ignora sin modelo"),
):
    """Devuelve el histórico reciente y el pronóstico.

    Con modelo entrenado cada punto futuro lleva su banda `[inferior, superior]`
    y la respuesta incluye las métricas medidas fuera de muestra. Sin artefacto
    servible se repite la última semana y `modelo_entrenado` es falso.
    """
    return demand_service.forecast(horizonte, historico, nivel)
