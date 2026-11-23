"""Offline-trained ETA inference. No fallback or classification outputs."""
import hashlib
import io
import logging
import math
import warnings
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline

from app.core.config import DELIVERY_DURATION_MODEL
from app.schemas import DeliveryDurationInput, DeliveryDurationOutput

MODEL_SHA256 = "4a698a3771aa50c0e945d24db434fa7b074b3a16823201128966ba54d804df20"
MODEL_SIZE = 326_117
MODEL_NAME = "HistGradientBoostingRegressor"
NUMERIC_FEATURES = ("precio", "flete", "peso_g", "n_items", "mes", "dia_semana", "mismo_estado")
STATE_FEATURES = ("estado_cliente", "estado_vendedor")
logger = logging.getLogger(__name__)


class ModelUnavailable(Exception):
    pass


class ArrivalDateOutOfRange(Exception):
    pass


@lru_cache(maxsize=1)
def _load_model(path: Path, fingerprint: tuple[int, int, int]) -> Pipeline:
    # Only deserialize the exact trusted bytes verified here, not a second read.
    if fingerprint[0] != MODEL_SIZE:
        raise ValueError("Tamaño de artefacto inesperado")
    with path.open("rb") as file:
        payload = file.read(MODEL_SIZE + 1)
    if hashlib.sha256(payload).hexdigest() != MODEL_SHA256:
        raise ValueError("El artefacto no coincide con el SHA-256 aprobado")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(io.BytesIO(payload))
    if not isinstance(model, Pipeline) or not isinstance(model.steps[-1][1], HistGradientBoostingRegressor):
        raise ValueError("Artefacto ETA incompatible")
    columns = {name: tuple(selected) for name, _, selected in model.named_steps["columntransformer"].transformers}
    if columns != {"cat": STATE_FEATURES, "num": NUMERIC_FEATURES}:
        raise ValueError("Las variables del artefacto no coinciden con el contrato ETA")
    return model


def model_row(data: DeliveryDurationInput) -> pd.DataFrame:
    date = data.fecha_compra
    return pd.DataFrame([{
        "precio": data.precio,
        "flete": data.flete,
        "peso_g": data.peso_g if data.peso_g > 0 else math.nan,
        "n_items": data.n_items,
        "mes": date.month if date else data.mes,
        "dia_semana": date.weekday() if date else data.dia_semana,
        "mismo_estado": int(data.estado_cliente == data.estado_vendedor),
        "estado_cliente": data.estado_cliente,
        "estado_vendedor": data.estado_vendedor,
    }])


def predict(data: DeliveryDurationInput) -> DeliveryDurationOutput:
    try:
        stat = DELIVERY_DURATION_MODEL.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        model = _load_model(DELIVERY_DURATION_MODEL, fingerprint)
        duration = float(model.predict(model_row(data))[0])
        if not math.isfinite(duration):
            raise ValueError("Predicción no finita")
        duration = max(duration, 0.)  # Same postprocessing as the experiment.
    except FileNotFoundError as error:
        raise ModelUnavailable from error
    except Exception as error:
        logger.warning("No fue posible cargar o ejecutar el modelo ETA", exc_info=True)
        raise ModelUnavailable from error

    try:
        arrival = data.fecha_compra + timedelta(days=math.floor(duration)) if data.fecha_compra else None
    except OverflowError as error:
        raise ArrivalDateOutOfRange from error
    return DeliveryDurationOutput(
        dias_entrega_estimados=duration,
        fecha_llegada_estimada=arrival,
        modelo=MODEL_NAME,
        experimental=True,
        mae_desarrollo_dias=4.60,
        mae_agosto_retrospectivo_dias=3.31,
    )
