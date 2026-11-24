"""Esquemas de entrada/salida de la API (Pydantic)."""
import re
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


BRAZIL_STATES = frozenset("AC AL AM AP BA CE DF ES GO MA MG MS MT PA PB PE PI PR RJ RN RO RR RS SC SE SP TO".split())


class DeliveryDurationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    precio: float = Field(ge=0, strict=True, description="Precio total en BRL")
    flete: float = Field(ge=0, strict=True, description="Flete total en BRL")
    peso_g: float = Field(ge=0, strict=True, description="Peso total en gramos; 0 indica peso desconocido")
    n_items: int = Field(ge=1, strict=True)
    estado_cliente: str
    estado_vendedor: str = Field(description="Estado representativo del origen del pedido")
    fecha_compra: date | None = Field(default=None, description="Fecha local de compra, YYYY-MM-DD")
    mes: int | None = Field(default=None, ge=1, le=12, strict=True)
    dia_semana: int | None = Field(default=None, ge=0, le=6, strict=True, description="0 = lunes")

    @field_validator("estado_cliente", "estado_vendedor")
    @classmethod
    def normalize_state(cls, value: str) -> str:
        state = value.strip().upper()
        if state not in BRAZIL_STATES:
            raise ValueError("Usa una sigla de estado de Brasil")
        return state

    @field_validator("fecha_compra", mode="before")
    @classmethod
    def validate_date_format(cls, value):
        if value is None or type(value) is date:
            return value
        if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            raise ValueError("Usa una fecha local en formato YYYY-MM-DD")
        return value

    @model_validator(mode="after")
    def validate_purchase_calendar(self):
        if self.fecha_compra is None:
            if self.mes is None or self.dia_semana is None:
                raise ValueError("Sin fecha de compra, indica el mes y el día de la semana")
        elif self.mes is not None or self.dia_semana is not None:
            raise ValueError("Indica la fecha de compra o el mes y día de la semana, no ambos")
        return self


class DeliveryDurationOutput(BaseModel):
    dias_entrega_estimados: float = Field(ge=0, allow_inf_nan=False)
    fecha_llegada_estimada: date | None
    modelo: str
    experimental: bool
    mae_desarrollo_dias: float
    mae_agosto_retrospectivo_dias: float


class ForecastPoint(BaseModel):
    fecha: str
    valor: float
    tipo: str  # "historico" | "pronostico"
    inferior: float | None = Field(
        None, description="Extremo inferior de la banda; nulo en los puntos históricos"
    )
    superior: float | None = Field(
        None, description="Extremo superior de la banda; nulo en los puntos históricos"
    )


class DemandMetrics(BaseModel):
    """Lo que se midió en el holdout reservado. Ver docs/evaluacion_demanda.md."""
    mae: float = Field(..., description="Error absoluto medio, en pedidos por día")
    mase: float = Field(..., description="Error escalado; adimensional")
    rel_mae: float = Field(
        ..., description="MAE relativo a repetir la última semana; por debajo de 1 la bate"
    )
    cobertura_80: float | None = Field(
        None, description="Proporción de días reales dentro de la banda del 80 %"
    )
    baseline_mae: float | None = None
    periodo: str = Field(..., description="Días evaluados, fuera de muestra")


class DemandOutput(BaseModel):
    serie: list[ForecastPoint]
    modelo_entrenado: bool
    modelo: str = Field("provisional", description="Especificación que produjo el pronóstico")
    nivel_confianza: float | None = Field(
        None, ge=0.5, lt=1, description="Nivel nominal de la banda; nulo sin modelo"
    )
    ancla: str | None = Field(
        None, description="Último día con datos: el pronóstico empieza al siguiente"
    )
    ventana: dict | None = Field(
        None, description="Tramo de histórico con el que se entrenó el modelo"
    )
    metricas: DemandMetrics | None = None


class ProductScore(BaseModel):
    categoria: str
    confianza: float = Field(..., ge=0, le=1)


class ProductOutput(BaseModel):
    categoria: str
    confianza: float
    top3: list[ProductScore]
    baja_confianza: bool = Field(
        False, description="True si la confianza no alcanza el umbral validado: revisar la categoría manualmente")
    umbral_confianza: float
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
