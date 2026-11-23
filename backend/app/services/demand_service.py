"""Pronóstico de la demanda diaria de pedidos (componente 2, series temporales).

Tres caminos, declarados siempre en la respuesta:

1. **Modelo entrenado.** Se reconstruye desde el artefacto con
   `demand_model.desempaquetar` y se publica el pronóstico con su banda y las
   métricas que se midieron en el holdout reservado.
2. **Sin artefacto o con un artefacto incompatible.** Se repite la última semana
   del histórico real de Olist y `modelo_entrenado` queda en falso. Un artefacto
   roto nunca produce un 500: produce este modo.
3. **Sin datos de Olist.** Serie sintética de demostración, que la interfaz
   señala como tal.
"""
import logging

import joblib
import numpy as np
import pandas as pd

from app.core.config import DEMAND_MODEL
from app.db import olist
from app.schemas import DemandMetrics, DemandOutput, ForecastPoint
from app.services.demand_calendar import calendar_exog
from app.services.demand_model import (
    BundleIncompatible, ajustar, desempaquetar, pronosticar,
)

log = logging.getLogger(__name__)

#: Cacheado por identidad del fichero: cambia el artefacto, se recarga solo.
_cache: tuple[tuple[int, int, int], object] | None = None


def _huella() -> tuple[int, int, int] | None:
    try:
        estado = DEMAND_MODEL.stat()
    except OSError:
        return None
    return (estado.st_size, estado.st_mtime_ns, estado.st_ino)


def _modelo():
    """(histórico, especificación, resultados, metadatos) o None si no hay artefacto servible.

    Los metadatos (ventana y métricas del holdout) se guardan en la caché junto
    al modelo: leer el artefacto en cada petición sería gratuito solo en apariencia.
    """
    global _cache
    huella = _huella()
    if huella is None:
        _cache = None
        return None
    if _cache is not None and _cache[0] == huella:
        return _cache[1]
    try:
        bundle = joblib.load(DEMAND_MODEL)
        serie, spec, resultados = desempaquetar(bundle)
        cargado = (serie, spec, resultados, {
            "window": bundle.get("window"),
            "metrics": bundle.get("metrics") or {},
            "calibration": bundle.get("calibration"),
        })
    except BundleIncompatible as error:
        log.warning("Artefacto de demanda inservible (%s); se usa el modo provisional", error)
        _cache = (huella, None)
        return None
    except Exception as error:  # fichero corrupto, pickle ilegible, dependencia ausente
        log.warning("No se pudo leer %s (%s); se usa el modo provisional", DEMAND_MODEL, error)
        _cache = (huella, None)
        return None
    _cache = (huella, cargado)
    return cargado


def _historico_vivo() -> pd.Series | None:
    """Serie diaria recalculada desde los CSV, si están disponibles."""
    if not olist.available("orders"):
        return None
    marcas = pd.to_datetime(olist.load("orders")["order_purchase_timestamp"], errors="coerce")
    marcas = marcas.dropna()
    if marcas.empty:
        return None
    serie = marcas.dt.floor("D").value_counts().sort_index().asfreq("D", fill_value=0)
    return serie.astype(float)


def _sintetico() -> pd.Series:
    indice = pd.date_range(end=pd.Timestamp.today().normalize(), periods=120, freq="D")
    generador = np.random.default_rng(0)
    valores = (150 + np.arange(120) * 0.8
               + 25 * np.sin(np.arange(120) * 2 * np.pi / 7)
               + generador.normal(0, 10, 120))
    return pd.Series(valores.clip(0), index=indice)


def _al_dia(serie: pd.Series, resultados, spec, bundle_fin: pd.Timestamp):
    """Extiende el modelo con los días reales posteriores al artefacto.

    En `sarimax`, `append` vuelve a filtrar con los mismos parámetros, así que
    incorpora datos nuevos sin reoptimizar. En `perfil` basta con recalcular la
    forma y el nivel sobre la serie ampliada, que cuesta menos de un milisegundo
    y no tiene optimizador que pueda desviarse.

    Solo se aplica si la continuación es limpia; ante cualquier duda se sigue
    con el histórico del artefacto.
    """
    vivo = _historico_vivo()
    if vivo is None:
        return serie, resultados
    nuevos = vivo.loc[vivo.index > bundle_fin]
    if nuevos.empty:
        return serie, resultados
    esperado = pd.date_range(bundle_fin + pd.Timedelta(days=1), nuevos.index[-1], freq="D")
    if not nuevos.index.equals(esperado) or (nuevos == 0).any():
        return serie, resultados
    try:
        completa = pd.concat([serie, nuevos])
        if spec.familia == "perfil":
            return completa, ajustar(completa, spec)

        from app.services.demand_model import transformar

        endog = pd.Series(transformar(nuevos.to_numpy(dtype=float), spec.transform),
                          index=nuevos.index)
        exog = calendar_exog(nuevos.index) if spec.exog else None
        return completa, resultados.append(endog, exog=exog)
    except Exception as error:
        log.warning("No se pudieron anexar %d días nuevos (%s)", len(nuevos), error)
        return serie, resultados


def _puntos(serie: pd.Series, ventana_historico: int) -> list[ForecastPoint]:
    recorte = serie.iloc[-ventana_historico:] if ventana_historico > 0 else serie.iloc[0:0]
    return [
        ForecastPoint(fecha=f"{fecha:%Y-%m-%d}", valor=round(float(valor), 2), tipo="historico")
        for fecha, valor in recorte.items()
    ]


def forecast(horizonte: int, ventana_historico: int = 90, nivel: float = 0.8) -> DemandOutput:
    """Histórico reciente más el pronóstico, con banda cuando hay modelo."""
    cargado = _modelo()

    if cargado is None:
        historico = _historico_vivo()
        if historico is None:
            historico = _sintetico()
        else:
            bundle_fin = historico.index[-1]
            historico = historico.loc[:bundle_fin]
        ultima_semana = historico.iloc[-7:].to_numpy()
        futuro = pd.date_range(historico.index[-1] + pd.Timedelta(days=1),
                               periods=horizonte, freq="D")
        serie = _puntos(historico, ventana_historico) + [
            ForecastPoint(fecha=f"{fecha:%Y-%m-%d}",
                          valor=round(float(max(ultima_semana[i % 7], 0)), 2),
                          tipo="pronostico")
            for i, fecha in enumerate(futuro)
        ]
        return DemandOutput(
            serie=serie, modelo_entrenado=False, modelo="provisional",
            ancla=f"{historico.index[-1]:%Y-%m-%d}",
        )

    historico, spec, resultados, meta = cargado
    historico, resultados = _al_dia(historico, resultados, spec, historico.index[-1])
    try:
        marco = pronosticar(resultados, historico, spec, horizonte, nivel,
                            calibracion=meta["calibration"])
    except Exception as error:
        log.warning("El modelo de demanda no pudo pronosticar (%s)", error)
        return forecast_provisional(horizonte, ventana_historico, historico)

    serie = _puntos(historico, ventana_historico) + [
        ForecastPoint(
            fecha=f"{fecha:%Y-%m-%d}", valor=round(float(fila["valor"]), 2), tipo="pronostico",
            inferior=round(float(fila["inferior"]), 2), superior=round(float(fila["superior"]), 2),
        )
        for fecha, fila in marco.iterrows()
    ]
    medidas = meta["metrics"]
    return DemandOutput(
        serie=serie,
        modelo_entrenado=True,
        modelo=spec.nombre + (" · banda calibrada" if meta["calibration"] else ""),
        nivel_confianza=nivel,
        ancla=f"{historico.index[-1]:%Y-%m-%d}",
        ventana=meta["window"],
        metricas=DemandMetrics(**medidas) if {"mae", "mase", "rel_mae", "periodo"} <= set(medidas)
        else None,
    )


def forecast_provisional(horizonte: int, ventana_historico: int, historico: pd.Series) -> DemandOutput:
    """Repetición de la última semana sobre un histórico ya cargado."""
    ultima_semana = historico.iloc[-7:].to_numpy()
    futuro = pd.date_range(historico.index[-1] + pd.Timedelta(days=1), periods=horizonte, freq="D")
    serie = _puntos(historico, ventana_historico) + [
        ForecastPoint(fecha=f"{fecha:%Y-%m-%d}",
                      valor=round(float(max(ultima_semana[i % 7], 0)), 2), tipo="pronostico")
        for i, fecha in enumerate(futuro)
    ]
    return DemandOutput(serie=serie, modelo_entrenado=False, modelo="provisional",
                        ancla=f"{historico.index[-1]:%Y-%m-%d}")
