"""Pronóstico de la demanda diaria de pedidos (series temporales)."""
import joblib
import numpy as np
import pandas as pd

from app.core.config import DEMAND_MODEL
from app.db import olist
from app.schemas import DemandOutput, ForecastPoint

_bundle = None


def _get_bundle():
    """El bundle contiene {'results': SARIMAXResults, 'history': pd.Series}."""
    global _bundle
    if _bundle is None and DEMAND_MODEL.exists():
        _bundle = joblib.load(DEMAND_MODEL)
    return _bundle


def _history() -> pd.Series:
    if olist.available("orders"):
        orders = olist.load("orders")
        s = (pd.to_datetime(orders["order_purchase_timestamp"])
             .dt.floor("D").value_counts().sort_index().asfreq("D", fill_value=0))
        return s.loc["2017-01-01":"2018-08-15"]
    # Serie sintética de ejemplo
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=120, freq="D")
    rng = np.random.default_rng(0)
    vals = 150 + np.arange(120) * 0.8 + 25 * np.sin(np.arange(120) * 2 * np.pi / 7) + rng.normal(0, 10, 120)
    return pd.Series(vals.clip(0), index=idx)


def forecast(horizonte: int, ventana_historico: int = 90) -> DemandOutput:
    bundle = _get_bundle()
    if bundle is not None:
        hist = bundle["history"]
        pred = bundle["results"].get_forecast(steps=horizonte).predicted_mean
        entrenado = True
    else:
        # Provisional: media móvil de 7 días con estacionalidad semanal ingenua
        hist = _history()
        last_week = hist.iloc[-7:].to_numpy()
        future_idx = pd.date_range(hist.index[-1] + pd.Timedelta(days=1), periods=horizonte, freq="D")
        pred = pd.Series([last_week[i % 7] for i in range(horizonte)], index=future_idx)
        entrenado = False

    serie = [ForecastPoint(fecha=d.strftime("%Y-%m-%d"), valor=round(float(v), 2), tipo="historico")
             for d, v in hist.iloc[-ventana_historico:].items()]
    serie += [ForecastPoint(fecha=d.strftime("%Y-%m-%d"), valor=round(max(float(v), 0), 2), tipo="pronostico")
              for d, v in pred.items()]
    return DemandOutput(serie=serie, modelo_entrenado=entrenado)
