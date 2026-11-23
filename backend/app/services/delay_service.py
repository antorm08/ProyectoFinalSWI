"""Predicción del riesgo de retraso de un pedido (clasificación binaria)."""
import joblib
import pandas as pd

from app.core.config import DELAY_MODEL
from app.schemas import DelayInput, DelayOutput

_model = None


def _get_model():
    global _model
    if _model is None and DELAY_MODEL.exists():
        _model = joblib.load(DELAY_MODEL)
    return _model


def _nivel(p: float) -> str:
    return "alto" if p >= 0.5 else "medio" if p >= 0.2 else "bajo"


def predict(data: DelayInput) -> DelayOutput:
    model = _get_model()
    row = pd.DataFrame([{
        **data.model_dump(),
        "mismo_estado": int(data.estado_cliente.upper() == data.estado_vendedor.upper()),
    }])
    row["estado_cliente"] = row["estado_cliente"].str.upper()
    row["estado_vendedor"] = row["estado_vendedor"].str.upper()
    # Igual que en preparación: cero significa peso desconocido, nunca peso real.
    row.loc[row["peso_g"].eq(0), "peso_g"] = float("nan")

    if model is not None:
        p = float(model.predict_proba(row)[0, 1])
        return DelayOutput(probabilidad_retraso=round(p, 4), nivel_riesgo=_nivel(p), modelo_entrenado=True)

    # Heurística provisional mientras no exista el modelo entrenado (ml/train_delay.py)
    p = 0.08
    p += 0.0 if row["mismo_estado"][0] else 0.07
    p += min(data.flete / max(data.precio, 1), 1) * 0.15
    p += 0.10 if data.dias_estimados < 10 else 0.0
    p += 0.08 if data.mes in (11, 12, 2, 3) else 0.0
    p = min(p, 0.95)
    return DelayOutput(probabilidad_retraso=round(p, 4), nivel_riesgo=_nivel(p), modelo_entrenado=False)
