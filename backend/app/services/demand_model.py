"""Capa de modelo del componente 2: especificación, ajuste, artefacto y pronóstico.

Vive en el backend porque es el servicio quien tiene que *reconstruir* el modelo
en cada arranque. Los scripts de `ml/` importan este mismo módulo, así que el
artefacto se verifica contra el código que de verdad lo va a servir.

**Por qué no se serializa el objeto de statsmodels.** El script original guardaba
`joblib.dump({"results": SARIMAXResults, ...})`, que arrastra el modelo entero y
queda atado a la versión exacta de la librería. Aquí se guardan solo los
parámetros ajustados y la especificación; el modelo se reconstruye con
`SARIMAX(...).smooth(params)`, que aplica el filtro de Kalman con esos parámetros
sin volver a optimizar. El artefacto es pequeño, legible y sobrevive a una
actualización de statsmodels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.services.demand_calendar import COLUMNAS as EXOG_COLUMNAS
from app.services.demand_calendar import calendar_exog

#: Versión del formato del artefacto. El servicio rechaza cualquier otra.
ESQUEMA = 2

#: Estacionalidad semanal.
PERIODO = 7

TRANSFORMACIONES = ("log1p", "identidad")

#: Días por bloque de la calibración empírica de la banda.
BLOQUE_DIAS = 7


class BundleIncompatible(Exception):
    """El artefacto no se puede usar: formato, columnas o contenido inesperados."""


@dataclass(frozen=True)
class Especificacion:
    """Todo lo que hace falta para reconstruir el modelo a partir de los parámetros."""
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, PERIODO)
    transform: str = "log1p"
    exog: bool = True

    def __post_init__(self) -> None:
        if self.transform not in TRANSFORMACIONES:
            raise ValueError(f"Transformación desconocida: {self.transform}")
        if len(self.order) != 3 or len(self.seasonal_order) != 4:
            raise ValueError("Órdenes mal formados")
        if any(v < 0 for v in (*self.order, *self.seasonal_order)):
            raise ValueError("Los órdenes no pueden ser negativos")

    @classmethod
    def desde_dict(cls, datos: dict) -> "Especificacion":
        return cls(
            order=tuple(datos["order"]),
            seasonal_order=tuple(datos["seasonal_order"]),
            transform=str(datos["transform"]),
            exog=bool(datos["exog"]),
        )

    def a_dict(self) -> dict:
        return {**asdict(self), "order": list(self.order), "seasonal_order": list(self.seasonal_order)}

    @property
    def nombre(self) -> str:
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        partes = [f"SARIMA({p},{d},{q})({P},{D},{Q})[{s}]"]
        if self.transform != "identidad":
            partes.append(self.transform)
        if self.exog:
            partes.append("calendario")
        return " · ".join(partes)


# --------------------------------------------------------------------------- #
# Transformación
# --------------------------------------------------------------------------- #
def transformar(valores: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        if np.any(valores < 0):
            raise ValueError("log1p no admite conteos negativos")
        return np.log1p(valores)
    return np.asarray(valores, dtype=float)


def destransformar(valores: np.ndarray, transform: str) -> np.ndarray:
    """Vuelve a pedidos por día.

    Con `log1p`, exponenciar la media en escala logarítmica devuelve la
    **mediana** de la distribución, no su media. Se publica esa mediana a
    propósito: es el valor típico de un día, y los extremos del intervalo se
    transforman exactamente porque la función es monótona, así que la cobertura
    de la banda sigue siendo la que se mide. Además nunca sale negativo, lo que
    hace innecesario el `max(v, 0)` que recortaba el servicio anterior.
    """
    if transform == "log1p":
        return np.expm1(np.asarray(valores, dtype=float))
    return np.asarray(valores, dtype=float)


# --------------------------------------------------------------------------- #
# Ajuste y reconstrucción
# --------------------------------------------------------------------------- #
def _exog(index: pd.DatetimeIndex, spec: Especificacion) -> pd.DataFrame | None:
    return calendar_exog(index) if spec.exog else None


def construir(serie: pd.Series, spec: Especificacion):
    """Modelo sin ajustar, sobre la serie ya transformada."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    if serie.index.freq is None:
        raise ValueError("La serie debe tener frecuencia diaria declarada")
    endog = pd.Series(transformar(serie.to_numpy(dtype=float), spec.transform), index=serie.index)
    return SARIMAX(
        endog,
        exog=_exog(serie.index, spec),
        order=spec.order,
        seasonal_order=spec.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )


def ajustar(serie: pd.Series, spec: Especificacion, maxiter: int = 200):
    """Optimiza los parámetros. Es la única parte costosa.

    Con el `maxiter` por defecto de statsmodels la optimización no converge en
    esta serie. Se sube el límite y la convergencia se consulta con
    `convergio()`, para que un ajuste no convergido no pase inadvertido.
    """
    return construir(serie, spec).fit(disp=False, maxiter=maxiter)


def convergio(resultados) -> bool:
    """True si el optimizador declaró convergencia. No es garantía de calidad."""
    return bool(getattr(resultados, "mle_retvals", {}).get("converged", False))


def reconstruir(serie: pd.Series, spec: Especificacion, params: np.ndarray):
    """Mismo estado que `ajustar`, sin volver a optimizar."""
    modelo = construir(serie, spec)
    params = np.asarray(params, dtype=float)
    if params.shape != (len(modelo.param_names),):
        raise BundleIncompatible(
            f"El artefacto trae {params.shape} parámetros y la especificación pide "
            f"{len(modelo.param_names)}"
        )
    return modelo.smooth(params)


def banda_calibrada(valores: np.ndarray, calibracion: dict, nivel: float):
    """Banda a partir de los errores observados en el backtesting, no de la teoría.

    El intervalo que publica statsmodels asume que el modelo es correcto y que
    los residuos se comportan; medido sobre Olist, ninguna especificación
    cumplía las dos cosas a la vez (ver docs/evaluacion_demanda.md). Aquí la
    banda se construye con los **cuantiles empíricos del error relativo**
    observado a cada distancia del origen, agrupados por bloques de
    `bloque_dias`. Así la cobertura es la medida, no la supuesta, y crece con
    el horizonte porque los errores crecen.

    Guarda los errores en bruto, no cuantiles precalculados, para poder servir
    cualquier nivel de confianza que se pida.
    """
    bloques = calibracion.get("errores_relativos")
    if not bloques:
        raise BundleIncompatible("La calibración no trae errores relativos")
    tam = int(calibracion.get("bloque_dias", BLOQUE_DIAS))
    if tam < 1:
        raise BundleIncompatible("Tamaño de bloque inválido en la calibración")
    bajo, alto = (1 - nivel) / 2, (1 + nivel) / 2
    cuantiles = []
    for errores in bloques:
        muestra = np.asarray(errores, dtype=float)
        if muestra.size == 0:
            raise BundleIncompatible("Un bloque de calibración está vacío")
        cuantiles.append((float(np.quantile(muestra, bajo)), float(np.quantile(muestra, alto))))
    indices = np.minimum(np.arange(len(valores)) // tam, len(cuantiles) - 1)
    inferior = np.array([valores[i] * (1 + cuantiles[b][0]) for i, b in enumerate(indices)])
    superior = np.array([valores[i] * (1 + cuantiles[b][1]) for i, b in enumerate(indices)])
    return np.maximum(inferior, 0.0), np.maximum(superior, inferior)


def pronosticar(
    resultados, serie: pd.Series, spec: Especificacion, steps: int, nivel: float = 0.8,
    calibracion: dict | None = None,
) -> pd.DataFrame:
    """Pronóstico y banda, en pedidos por día, con índice de fechas futuras.

    Con `calibracion` la banda sale de los errores medidos; sin ella, del
    intervalo teórico del modelo.
    """
    if steps < 1:
        raise ValueError("steps debe ser al menos 1")
    if not 0.5 <= nivel < 1:
        raise ValueError("El nivel de confianza debe estar en [0,5, 1)")
    futuro = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    salida = resultados.get_forecast(steps=steps, exog=_exog(futuro, spec))
    valor = destransformar(salida.predicted_mean.to_numpy(), spec.transform)
    if calibracion:
        inferior, superior = banda_calibrada(valor, calibracion, nivel)
    else:
        banda = salida.conf_int(alpha=1 - nivel).to_numpy()
        inferior = destransformar(banda[:, 0], spec.transform)
        superior = destransformar(banda[:, 1], spec.transform)
    marco = pd.DataFrame({"valor": valor, "inferior": inferior, "superior": superior}, index=futuro)
    marco["inferior"] = marco["inferior"].clip(lower=0.0)
    return marco


# --------------------------------------------------------------------------- #
# Artefacto
# --------------------------------------------------------------------------- #
CLAVES = ("schema", "spec", "params", "history", "exog_columns")


def empaquetar(serie: pd.Series, spec: Especificacion, resultados,
               calibracion: dict | None = None, **extra: Any) -> dict:
    """Bundle listo para `joblib.dump`, con la comprobación de ida y vuelta dentro."""
    comprobacion = pronosticar(resultados, serie, spec, steps=PERIODO, nivel=0.8,
                               calibracion=calibracion)
    return {
        "schema": ESQUEMA,
        "spec": spec.a_dict(),
        "params": np.asarray(resultados.params, dtype=float),
        "history": serie,
        "exog_columns": list(EXOG_COLUMNAS) if spec.exog else [],
        "calibration": calibracion,
        "forecast_check": {
            "steps": PERIODO,
            "nivel": 0.8,
            "valor": comprobacion["valor"].tolist(),
            "inferior": comprobacion["inferior"].tolist(),
            "superior": comprobacion["superior"].tolist(),
        },
        **extra,
    }


def desempaquetar(bundle: Any) -> tuple[pd.Series, Especificacion, Any]:
    """Valida el bundle y devuelve (histórico, especificación, resultados).

    Cualquier problema sale como `BundleIncompatible`, nunca como un 500: el
    servicio lo traduce a modo provisional.
    """
    if not isinstance(bundle, dict):
        raise BundleIncompatible("El artefacto no es un diccionario")
    if bundle.get("schema") != ESQUEMA:
        raise BundleIncompatible(
            f"Esquema {bundle.get('schema')!r}; este servicio sirve el {ESQUEMA}"
        )
    faltan = [c for c in CLAVES if c not in bundle]
    if faltan:
        raise BundleIncompatible(f"Faltan claves en el artefacto: {faltan}")
    try:
        spec = Especificacion.desde_dict(bundle["spec"])
    except (KeyError, TypeError, ValueError) as error:
        raise BundleIncompatible(f"Especificación inválida: {error}") from error
    calibracion = bundle.get("calibration")
    if calibracion is not None:
        if not isinstance(calibracion, dict) or not calibracion.get("errores_relativos"):
            raise BundleIncompatible("La calibración del artefacto está mal formada")
        if any(len(bloque) == 0 for bloque in calibracion["errores_relativos"]):
            raise BundleIncompatible("La calibración trae algún bloque vacío")
    esperadas = list(EXOG_COLUMNAS) if spec.exog else []
    if list(bundle["exog_columns"]) != esperadas:
        raise BundleIncompatible(
            f"Columnas exógenas {bundle['exog_columns']!r}; se esperaban {esperadas!r}"
        )
    serie = bundle["history"]
    if not isinstance(serie, pd.Series) or not isinstance(serie.index, pd.DatetimeIndex):
        raise BundleIncompatible("El histórico del artefacto no es una serie con fechas")
    if serie.empty or serie.isna().any():
        raise BundleIncompatible("El histórico del artefacto está vacío o tiene huecos")
    if serie.index.freq is None:
        serie = serie.asfreq("D")
        if serie.isna().any():
            raise BundleIncompatible("El histórico no es una serie diaria continua")
    try:
        resultados = reconstruir(serie, spec, bundle["params"])
    except BundleIncompatible:
        raise
    except Exception as error:  # statsmodels, numpy, calendario fuera de rango
        raise BundleIncompatible(f"No se pudo reconstruir el modelo: {error}") from error
    return serie, spec, resultados


def verificar(bundle: dict, tolerancia: float = 1e-6) -> dict:
    """Comprueba que el artefacto recargado reproduce el pronóstico guardado.

    Mismo criterio que `ml/train_products.export()` con TorchScript: si el
    artefacto no reproduce lo que se midió, no se activa.
    """
    guardado = bundle.get("forecast_check")
    if not guardado:
        raise BundleIncompatible("El artefacto no trae comprobación de ida y vuelta")
    serie, spec, resultados = desempaquetar(bundle)
    vuelta = pronosticar(resultados, serie, spec, steps=guardado["steps"], nivel=guardado["nivel"],
                         calibracion=bundle.get("calibration"))
    desvios = {}
    for columna in ("valor", "inferior", "superior"):
        desvio = float(np.max(np.abs(vuelta[columna].to_numpy() - np.asarray(guardado[columna]))))
        desvios[columna] = desvio
        if desvio > tolerancia:
            raise BundleIncompatible(
                f"El artefacto recargado no reproduce {columna}: desvío {desvio:.3g}"
            )
    return {"tolerancia": tolerancia, "desvio_maximo": desvios}
