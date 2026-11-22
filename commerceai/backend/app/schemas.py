"""Esquemas de entrada/salida de la API (Pydantic)."""
from pydantic import BaseModel, Field


class DelayInput(BaseModel):
    precio: float = Field(..., ge=0, description="Precio total de los ítems (BRL)")
    flete: float = Field(..., ge=0, description="Coste de envío (BRL)")
    peso_g: float = Field(..., ge=0, description="Peso del producto en gramos")
    n_items: int = Field(1, ge=1)
    dias_estimados: int = Field(..., ge=0, description="Días prometidos de entrega")
    estado_cliente: str = Field(..., min_length=2, max_length=2, examples=["SP"])
    estado_vendedor: str = Field(..., min_length=2, max_length=2, examples=["SP"])
    mes: int = Field(..., ge=1, le=12)
    dia_semana: int = Field(..., ge=0, le=6, description="0 = lunes")


class DelayOutput(BaseModel):
    probabilidad_retraso: float
    nivel_riesgo: str
    modelo_entrenado: bool


class ForecastPoint(BaseModel):
    fecha: str
    valor: float
    tipo: str  # "historico" | "pronostico"


class DemandOutput(BaseModel):
    serie: list[ForecastPoint]
    modelo_entrenado: bool


class ProductOutput(BaseModel):
    categoria: str
    confianza: float
    top3: list[dict]
    modelo_entrenado: bool


class SellerMetrics(BaseModel):
    n_pedidos: float
    facturacion: float
    nota_media: float
    tasa_retraso: float
    dias_entrega_medio: float


class SellerClusterOutput(BaseModel):
    cluster: int
    etiqueta: str
    modelo_entrenado: bool


class SummaryInput(BaseModel):
    texto: str = Field(..., min_length=20)
    n_oraciones: int = Field(3, ge=1, le=15)


class SummaryOutput(BaseModel):
    resumen: str
    oraciones_originales: int
    oraciones_resumen: int
