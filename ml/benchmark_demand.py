"""2. Catálogo de algoritmos para el pronóstico de demanda, medido en el mismo banco.

`docs/evaluacion_demanda.md` dejó abierta una pregunta incómoda: una media por
día de la semana pronostica mejor que cualquier SARIMA probado. Este script la
investiga en serio, con el **mismo protocolo** que la evaluación (los mismos
ocho orígenes, el mismo horizonte y el mismo baseline) para que las cifras sean
comparables fila a fila.

El diagnóstico que guía el catálogo: el error del modelo activo es sobre todo
error de **nivel**, no de forma. Su sesgo pasa de +1,5 pedidos/día en la primera
semana del horizonte a +35,6 en la cuarta, porque extrapola la tendencia de
crecimiento de 2017 más allá de donde se sostiene. De ahí que el catálogo
insista en métodos amortiguados, en separar el nivel de la forma semanal y en
combinar.

**El holdout no se toca.** Este script solo usa los orígenes de desarrollo: su
cometido es ordenar candidatos, no certificar ninguno.

Salidas (no sobrescribe):
  docs/resultados_demanda_algoritmos/metricas.json
  docs/resultados_demanda_algoritmos/*.png

Uso:
  python -m ml.benchmark_demand
  python -m ml.benchmark_demand --output /tmp/prueba --solo rapidos
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .common import ROOT
    from .demand_series import (
        PERIODO, ahora_utc, code_hashes, coverage_window, csv_hashes, daily_orders,
        mase, metrics, rel_mae, rolling_origin, seasonal_naive, versiones, weekday_mean,
    )
else:
    from common import ROOT
    from demand_series import (
        PERIODO, ahora_utc, code_hashes, coverage_window, csv_hashes, daily_orders,
        mase, metrics, rel_mae, rolling_origin, seasonal_naive, versiones, weekday_mean,
    )

from app.services.demand_calendar import calendar_exog
from app.services.demand_model import Especificacion, ajustar, pronosticar

DEFAULT_OUTPUT = ROOT / "docs" / "resultados_demanda_algoritmos"
HORIZONTE = 28
N_ORIGENES = 8
PASO = 14
HOLDOUT = 28
BASELINE = "naive_estacional"

#: Especificación activa hoy, como referencia de la fila a batir.
SPEC_ACTIVA = Especificacion(order=(2, 1, 2), seasonal_order=(1, 1, 1, PERIODO),
                             transform="identidad", exog=True)

CODIGO = ("ml/benchmark_demand.py", "ml/demand_series.py",
          "backend/app/services/demand_model.py")


def _futuro(train: pd.Series, steps: int) -> pd.DatetimeIndex:
    return pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")


def _serie(valores, indice) -> pd.Series:
    return pd.Series(np.asarray(valores, dtype=float), index=indice, name="pronostico")


# --------------------------------------------------------------------------- #
# Familia 1: perfil semanal local. Lo que hoy gana, y sus variantes.
# --------------------------------------------------------------------------- #
def media_dia_semana(train, steps, semanas=4):
    return weekday_mean(train, steps, semanas=semanas)


def mediana_dia_semana(train, steps, semanas=8):
    """Mediana en vez de media: una semana atípica no arrastra el perfil."""
    reciente = train.iloc[-semanas * PERIODO:]
    perfil = reciente.groupby(reciente.index.day_of_week).median()
    indice = _futuro(train, steps)
    return _serie(perfil.reindex(indice.day_of_week).to_numpy(), indice)


def ewma_dia_semana(train, steps, semanas=12, alfa=0.35):
    """Perfil por día de la semana con peso exponencial: lo reciente manda."""
    reciente = train.iloc[-semanas * PERIODO:]
    perfil = {}
    for dia, grupo in reciente.groupby(reciente.index.day_of_week):
        valores = grupo.to_numpy(dtype=float)[::-1]          # del más nuevo al más viejo
        pesos = alfa * (1 - alfa) ** np.arange(len(valores))
        perfil[dia] = float(np.sum(valores * pesos) / np.sum(pesos))
    indice = _futuro(train, steps)
    return _serie([perfil[d] for d in indice.day_of_week], indice)


def perfil_por_nivel(train, steps, semanas_perfil=12, dias_nivel=28):
    """Separa **forma** y **nivel**, que es donde está el error.

    La forma semanal es estable y se estima con mucha historia; el nivel cambia
    y se estima con poca. El producto de ambos evita que una forma antigua
    arrastre un nivel viejo, y viceversa.
    """
    largo = train.iloc[-semanas_perfil * PERIODO:]
    forma = largo.groupby(largo.index.day_of_week).mean() / largo.mean()
    nivel = float(train.iloc[-dias_nivel:].mean())
    indice = _futuro(train, steps)
    return _serie(nivel * forma.reindex(indice.day_of_week).to_numpy(), indice)


def perfil_por_nivel_amortiguado(train, steps, semanas_perfil=12, dias_nivel=28, phi=0.9):
    """Igual, con una deriva amortiguada: la tendencia reciente se apaga."""
    base = perfil_por_nivel(train, steps, semanas_perfil, dias_nivel)
    reciente = train.iloc[-2 * dias_nivel:]
    deriva = (float(reciente.iloc[-dias_nivel:].mean()) - float(reciente.iloc[:dias_nivel].mean())) / dias_nivel
    pasos = np.arange(1, steps + 1)
    acumulado = deriva * np.cumsum(phi ** pasos)
    return _serie(np.maximum(base.to_numpy() + acumulado, 0), base.index)


# --------------------------------------------------------------------------- #
# Familia 2: estadística clásica
# --------------------------------------------------------------------------- #
def theta(train, steps):
    """Método Theta, ganador de la competición M3 y referencia obligada."""
    from statsmodels.tsa.forecasting.theta import ThetaModel

    modelo = ThetaModel(train, period=PERIODO, deseasonalize=True, method="additive").fit()
    return _serie(modelo.forecast(steps).to_numpy(), _futuro(train, steps))


def ets_amortiguado(train, steps):
    """Holt-Winters con tendencia amortiguada: el remedio clásico al sesgo largo."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    ajuste = ExponentialSmoothing(
        train, trend="add", damped_trend=True, seasonal="add", seasonal_periods=PERIODO,
        initialization_method="estimated",
    ).fit()
    return _serie(ajuste.forecast(steps).to_numpy(), _futuro(train, steps))


def ets_sin_tendencia(train, steps):
    """Sin término de tendencia: no puede extrapolar lo que no hay."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    ajuste = ExponentialSmoothing(
        train, trend=None, seasonal="add", seasonal_periods=PERIODO,
        initialization_method="estimated",
    ).fit()
    return _serie(ajuste.forecast(steps).to_numpy(), _futuro(train, steps))


def stl_arima(train, steps):
    """STL desestacionaliza y un ARIMA sin estacionalidad pronostica el resto."""
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.forecasting.stl import STLForecast

    ajuste = STLForecast(train, ARIMA, model_kwargs={"order": (1, 1, 1), "trend": "n"},
                         period=PERIODO, robust=True).fit()
    return _serie(np.asarray(ajuste.forecast(steps)), _futuro(train, steps))


def sarimax_activo(train, steps):
    """El modelo que hoy sirve la aplicación."""
    resultados = ajustar(train, SPEC_ACTIVA)
    return pronosticar(resultados, train, SPEC_ACTIVA, steps, 0.8)["valor"]


def arima_fourier(train, steps, armonicos=2):
    """Regresión armónica dinámica: la semana entra como senos y cosenos."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    def terminos(indice):
        paso = np.arange(len(indice)) if indice.equals(train.index) else (
            len(train) + np.arange(len(indice)))
        columnas = {}
        for k in range(1, armonicos + 1):
            columnas[f"sin{k}"] = np.sin(2 * np.pi * k * paso / PERIODO)
            columnas[f"cos{k}"] = np.cos(2 * np.pi * k * paso / PERIODO)
        marco = pd.DataFrame(columnas, index=indice)
        return pd.concat([marco, calendar_exog(indice)], axis=1)

    futuro = _futuro(train, steps)
    modelo = SARIMAX(train, exog=terminos(train.index), order=(2, 1, 2),
                     enforce_stationarity=False, enforce_invertibility=False)
    ajuste = modelo.fit(disp=False, maxiter=200)
    salida = ajuste.get_forecast(steps=steps, exog=terminos(futuro))
    return _serie(salida.predicted_mean.to_numpy(), futuro)


# --------------------------------------------------------------------------- #
# Familia 3: aprendizaje automático sobre calendario
# --------------------------------------------------------------------------- #
def _rasgos(indice: pd.DatetimeIndex, origen: pd.Timestamp) -> pd.DataFrame:
    dias = (indice - origen).days.to_numpy(dtype=float)
    marco = pd.DataFrame({"t": dias}, index=indice)
    for dia in range(PERIODO):
        marco[f"dow{dia}"] = (indice.day_of_week == dia).astype(float)
    for k in (1, 2):
        marco[f"sin{k}"] = np.sin(2 * np.pi * k * dias / PERIODO)
        marco[f"cos{k}"] = np.cos(2 * np.pi * k * dias / PERIODO)
    return pd.concat([marco, calendar_exog(indice)], axis=1)


def ridge_calendario(train, steps, dias=180, con_tendencia=True):
    """Regresión lineal regularizada sobre calendario; todo conocido de antemano."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    reciente = train.iloc[-dias:]
    origen = reciente.index[0]
    X = _rasgos(reciente.index, origen)
    futuro = _futuro(train, steps)
    Xf = _rasgos(futuro, origen)
    if not con_tendencia:
        X, Xf = X.drop(columns="t"), Xf.drop(columns="t")
    modelo = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, reciente.to_numpy())
    return _serie(np.maximum(modelo.predict(Xf), 0), futuro)


def hgb_calendario(train, steps, dias=365):
    """Árboles sobre los mismos rasgos: no pueden extrapolar la tendencia.

    Esa incapacidad, que suele ser un defecto, aquí amortigua por construcción.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    reciente = train.iloc[-dias:]
    origen = reciente.index[0]
    modelo = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_depth=4, l2_regularization=1.0,
        random_state=42,
    ).fit(_rasgos(reciente.index, origen), reciente.to_numpy())
    futuro = _futuro(train, steps)
    return _serie(np.maximum(modelo.predict(_rasgos(futuro, origen)), 0), futuro)


# --------------------------------------------------------------------------- #
# Familia 4: combinaciones
# --------------------------------------------------------------------------- #
def combinacion_mediana(train, steps):
    """Mediana de tres familias distintas. Combinar es la defensa barata."""
    miembros = [perfil_por_nivel(train, steps), theta(train, steps),
                ets_amortiguado(train, steps)]
    return _serie(np.median([m.to_numpy() for m in miembros], axis=0), miembros[0].index)


def combinacion_por_horizonte(train, steps):
    """Naive la primera semana, perfil por nivel después.

    El diagnóstico dice que el naive gana en la semana 1 (MAE 29,6 frente a
    35,8) y pierde a partir de la segunda. Esto lo aprovecha explícitamente.
    """
    corto = seasonal_naive(train, steps)
    largo = perfil_por_nivel(train, steps)
    mezcla = corto.to_numpy().copy()
    if steps > PERIODO:
        mezcla[PERIODO:] = largo.to_numpy()[PERIODO:]
    return _serie(mezcla, corto.index)


CATALOGO = {
    BASELINE: (seasonal_naive, "baseline", "Repetir la última semana"),
    "media_dia_semana_4s": (lambda t, s: media_dia_semana(t, s, 4), "perfil",
                            "Media por día de la semana, 4 semanas (la del informe)"),
    "media_dia_semana_8s": (lambda t, s: media_dia_semana(t, s, 8), "perfil",
                            "Media por día de la semana, 8 semanas"),
    "mediana_dia_semana_8s": (mediana_dia_semana, "perfil", "Mediana por día de la semana"),
    "ewma_dia_semana": (ewma_dia_semana, "perfil", "Perfil con peso exponencial"),
    "perfil_por_nivel": (perfil_por_nivel, "perfil", "Forma semanal larga x nivel reciente"),
    "perfil_por_nivel_amort": (perfil_por_nivel_amortiguado, "perfil",
                               "Igual, con deriva amortiguada"),
    "theta": (theta, "clasica", "Método Theta"),
    "ets_amortiguado": (ets_amortiguado, "clasica", "Holt-Winters con tendencia amortiguada"),
    "ets_sin_tendencia": (ets_sin_tendencia, "clasica", "Holt-Winters sin tendencia"),
    "stl_arima": (stl_arima, "clasica", "STL robusto + ARIMA(1,1,1)"),
    "arima_fourier": (arima_fourier, "clasica", "ARIMA(2,1,2) con armónicos de Fourier"),
    "sarimax_activo": (sarimax_activo, "clasica", "SARIMA(2,1,2)(1,1,1)[7] + calendario (activo)"),
    "ridge_calendario": (ridge_calendario, "aprendizaje", "Ridge sobre calendario y tendencia"),
    "ridge_sin_tendencia": (lambda t, s: ridge_calendario(t, s, con_tendencia=False),
                            "aprendizaje", "Ridge sobre calendario, sin tendencia"),
    "hgb_calendario": (hgb_calendario, "aprendizaje", "Gradient boosting sobre calendario"),
    "combinacion_mediana": (combinacion_mediana, "combinacion", "Mediana de tres familias"),
    "combinacion_horizonte": (combinacion_por_horizonte, "combinacion",
                              "Naive la semana 1, perfil por nivel después"),
}

RAPIDOS = {k for k, (_, familia, _) in CATALOGO.items() if familia != "clasica"} | {BASELINE}


def evaluar(nombre, funcion, particiones, errores=None) -> dict:
    filas, por_semana = [], {}
    absolutos = []
    inicio = time.perf_counter()
    for train, test in particiones:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = funcion(train, len(test))
        pred = pd.Series(np.asarray(pred, dtype=float), index=test.index)
        filas.append({
            **metrics(test, pred),
            "mase": mase(test, pred, train),
            "rel_mae": rel_mae(test, pred, train),
            "test_inicio": str(test.index[0].date()),
        })
        error = pred.to_numpy() - test.to_numpy()
        absolutos.append(np.abs(error))
        for semana in range(len(test) // PERIODO):
            tramo = slice(semana * PERIODO, (semana + 1) * PERIODO)
            por_semana.setdefault(semana + 1, []).extend(error[tramo])
    if errores is not None:
        errores[nombre] = absolutos
    relativos = [f["rel_mae"] for f in filas]
    return {
        "mae": float(np.mean([f["mae"] for f in filas])),
        "rmse": float(np.mean([f["rmse"] for f in filas])),
        "smape_pct": float(np.mean([f["smape_pct"] for f in filas])),
        "mase": float(np.mean([f["mase"] for f in filas])),
        "rel_mae": float(np.mean(relativos)),
        "rel_mae_mediana": float(np.median(relativos)),
        "rel_mae_sd": float(np.std(relativos, ddof=1)),
        "rel_mae_min": float(np.min(relativos)),
        "rel_mae_max": float(np.max(relativos)),
        "gana_al_baseline": sum(1 for v in relativos if v < 1),
        "bias": float(np.mean([f["bias"] for f in filas])),
        "mae_por_semana": {str(k): float(np.mean(np.abs(v))) for k, v in sorted(por_semana.items())},
        "bias_por_semana": {str(k): float(np.mean(v)) for k, v in sorted(por_semana.items())},
        "segundos_por_origen": round((time.perf_counter() - inicio) / len(particiones), 3),
        "origenes_detalle": filas,
    }


SEMILLA = 42
REPLICAS = 5000


def bootstrap_pareado(errores: dict, nombres: tuple[str, ...], n_origenes: int) -> list[dict]:
    """Compara candidatos remuestreando **orígenes**, no días.

    Los 28 días de una misma ventana comparten el mismo estado de la serie y el
    mismo ajuste: tratarlos como observaciones independientes estrecharía los
    intervalos de forma artificial.
    """
    generador = np.random.default_rng(SEMILLA)
    por_dia = {k: np.concatenate(v) for k, v in errores.items()}
    etiqueta = np.concatenate([[k] * HORIZONTE for k in range(n_origenes)])
    salida = []
    for candidato, referencia in nombres:
        if candidato not in por_dia or referencia not in por_dia:
            continue
        diferencia = por_dia[candidato] - por_dia[referencia]
        por_origen = np.array([diferencia[etiqueta == k].mean() for k in range(n_origenes)])
        muestras = generador.choice(por_origen, size=(REPLICAS, n_origenes),
                                    replace=True).mean(axis=1)
        salida.append({
            "candidato": candidato,
            "referencia": referencia,
            "delta_mae": round(float(por_origen.mean()), 3),
            "ic95": [round(float(x), 3) for x in np.percentile(muestras, [2.5, 97.5])],
            "p_es_mejor": round(float((muestras < 0).mean()), 3),
            "significativo": bool(np.percentile(muestras, 97.5) < 0),
        })
    return salida


#: Rejillas de sensibilidad. No sirven para elegir el mejor punto —eso sería
#: optimizar contra el propio banco— sino para ver si la ventaja de la familia
#: depende de acertar un parámetro o aguanta en una región ancha.
REJILLAS = {
    "perfil_por_nivel": {
        "parametros": ["semanas_forma", "dias_nivel"],
        "valores": [(sp, dn) for sp in (6, 8, 12, 16, 26) for dn in (14, 21, 28, 42, 56)],
        "funcion": lambda sp, dn: (lambda t, s: perfil_por_nivel(t, s, sp, dn)),
    },
    "media_dia_semana": {
        "parametros": ["semanas"],
        "valores": [(n,) for n in (2, 3, 4, 6, 8, 10, 12, 16)],
        "funcion": lambda n: (lambda t, s: media_dia_semana(t, s, n)),
    },
    "ewma_dia_semana": {
        "parametros": ["alfa"],
        "valores": [(a,) for a in (0.15, 0.25, 0.35, 0.5, 0.7)],
        "funcion": lambda a: (lambda t, s: ewma_dia_semana(t, s, 12, a)),
    },
}


def sensibilidad(particiones) -> dict:
    """relMAE de cada familia del perfil al mover sus parámetros."""
    salida = {}
    for nombre, rejilla in REJILLAS.items():
        puntos = []
        for valores in rejilla["valores"]:
            datos = evaluar(nombre, rejilla["funcion"](*valores), particiones)
            puntos.append({
                **dict(zip(rejilla["parametros"], valores)),
                "rel_mae": round(datos["rel_mae"], 4),
                "rel_mae_mediana": round(datos["rel_mae_mediana"], 4),
                "gana_al_baseline": datos["gana_al_baseline"],
            })
        relativos = [p["rel_mae"] for p in puntos]
        salida[nombre] = {
            "parametros": rejilla["parametros"],
            "puntos": puntos,
            "mejor": min(relativos),
            "peor": max(relativos),
            "mediana": float(np.median(relativos)),
            "puntos_bajo_090": sum(1 for v in relativos if v < 0.90),
            "total": len(relativos),
        }
    return salida


def render_plots(directorio: Path, resultados: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nombres = []

    def guardar(nombre, fig):
        ruta = directorio / f"{nombre}.png"
        fig.savefig(ruta, dpi=130, bbox_inches="tight")
        plt.close(fig)
        nombres.append(ruta.name)

    orden = sorted(resultados, key=lambda k: resultados[k]["rel_mae"])
    colores = {"baseline": "#0f172a", "perfil": "#2563eb", "clasica": "#f59e0b",
               "aprendizaje": "#10b981", "combinacion": "#a855f7"}
    fig, ax = plt.subplots(figsize=(9, 0.33 * len(orden) + 1.4))
    ax.barh(range(len(orden)), [resultados[k]["rel_mae"] for k in orden],
            color=[colores[CATALOGO[k][1]] for k in orden])
    ax.axvline(1.0, color="#0f172a", ls="--", lw=1)
    ax.set_yticks(range(len(orden))); ax.set_yticklabels(orden, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("MAE relativo a repetir la última semana")
    ax.set_title("Catálogo de algoritmos · 8 orígenes, horizonte 28 días")
    guardar("ranking_rel_mae", fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for nombre in orden[:6]:
        semanas = resultados[nombre]["mae_por_semana"]
        ax.plot(list(semanas), list(semanas.values()), marker="o", ms=4, lw=1.2, label=nombre)
    ax.set_xlabel("semana del horizonte"); ax.set_ylabel("MAE (pedidos/día)")
    ax.set_title("Dónde se gana y se pierde: MAE por semana del horizonte")
    ax.legend(fontsize=7)
    guardar("mae_por_semana", fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for nombre in orden[:6]:
        semanas = resultados[nombre]["bias_por_semana"]
        ax.plot(list(semanas), list(semanas.values()), marker="o", ms=4, lw=1.2, label=nombre)
    ax.axhline(0, color="#0f172a", lw=0.8)
    ax.set_xlabel("semana del horizonte"); ax.set_ylabel("sesgo (pedidos/día)")
    ax.set_title("Sesgo por semana del horizonte: quién extrapola de más")
    ax.legend(fontsize=7)
    guardar("sesgo_por_semana", fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for nombre in orden:
        datos = resultados[nombre]
        ax.scatter(datos["segundos_por_origen"], datos["rel_mae"],
                   color=colores[CATALOGO[nombre][1]], s=28)
        ax.annotate(nombre, (datos["segundos_por_origen"], datos["rel_mae"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.axhline(1.0, color="#0f172a", ls="--", lw=1)
    ax.set_xlabel("segundos por origen (escala logarítmica)")
    ax.set_ylabel("MAE relativo")
    ax.set_title("Coste frente a precisión")
    guardar("coste_vs_precision", fig)
    return nombres


def run(output: Path, *, solo: str | None = None) -> dict:
    inicio = time.perf_counter()
    arranque = ahora_utc()
    output = Path(output)
    destino = output / "metricas.json"
    if destino.exists():
        raise FileExistsError(f"{destino} ya existe; no se sobrescriben resultados")

    ventana = coverage_window(daily_orders())
    particiones = rolling_origin(ventana.serie, n_origenes=N_ORIGENES, horizonte=HORIZONTE,
                                 paso=PASO, holdout=HOLDOUT)
    print(f"Ventana {ventana.inicio.date()} a {ventana.fin.date()} · "
          f"{len(particiones)} orígenes · horizonte {HORIZONTE} días")
    print("El holdout final no se toca: este banco solo ordena candidatos.\n")

    elegidos = [k for k in CATALOGO if solo != "rapidos" or k in RAPIDOS]
    resultados, errores = {}, {}
    for nombre in elegidos:
        funcion = CATALOGO[nombre][0]
        try:
            resultados[nombre] = evaluar(nombre, funcion, particiones, errores)
        except Exception as error:
            print(f"  {nombre:24s} FALLÓ: {str(error)[:90]}")
            continue
        datos = resultados[nombre]
        print(f"  {nombre:24s} MAE {datos['mae']:6.2f}  relMAE {datos['rel_mae']:.3f}  "
              f"mediana {datos['rel_mae_mediana']:.3f}  gana {datos['gana_al_baseline']}/"
              f"{len(particiones)}  sesgo {datos['bias']:+6.1f}  "
              f"{datos['segundos_por_origen']:6.3f}s")

    orden = sorted(resultados, key=lambda k: resultados[k]["rel_mae"])
    print(f"\nMejor: {orden[0]} (relMAE {resultados[orden[0]]['rel_mae']:.3f})")
    activo = resultados.get("sarimax_activo", {}).get("rel_mae")
    if activo:
        mejora = (activo - resultados[orden[0]]["rel_mae"]) / activo * 100
        print(f"Frente al modelo activo ({activo:.3f}): {mejora:+.1f} % de error relativo")

    comparaciones = [(c, BASELINE) for c in orden if c != BASELINE]
    comparaciones += [(c, "sarimax_activo") for c in orden[:5] if c != "sarimax_activo"]
    comparaciones += [(orden[0], c) for c in orden[1:4]]
    pruebas = bootstrap_pareado(errores, tuple(comparaciones), len(particiones))
    print(f"\nBootstrap pareado sobre orígenes ({REPLICAS} réplicas, semilla {SEMILLA}):")
    for fila in pruebas:
        if fila["referencia"] != "sarimax_activo":
            continue
        marca = "significativo" if fila["significativo"] else "no significativo"
        print(f"  {fila['candidato']:24s} vs activo  Δ {fila['delta_mae']:+6.2f}  "
              f"IC [{fila['ic95'][0]:+6.2f}, {fila['ic95'][1]:+6.2f}]  "
              f"P {fila['p_es_mejor']:.3f}  {marca}")

    print("\nSensibilidad de la familia del perfil a sus parámetros…")
    sensible = sensibilidad(particiones)
    for nombre, datos in sensible.items():
        print(f"  {nombre:24s} relMAE de {datos['mejor']:.3f} a {datos['peor']:.3f}; "
              f"{datos['puntos_bajo_090']}/{datos['total']} puntos por debajo de 0,90")

    output.mkdir(parents=True, exist_ok=True)
    graficas = render_plots(output, resultados)
    informe = {
        "started_at_utc": arranque,
        "protocol": {
            "target": "pedidos creados por día, cualquier estado",
            "ventana": ventana.summary()["window"],
            "origenes": len(particiones),
            "horizonte_dias": HORIZONTE,
            "paso_dias": PASO,
            "baseline": BASELINE,
            "holdout": "reservado y no utilizado en este banco",
        },
        "versions": versiones(),
        "source_csv_sha256": csv_hashes(),
        "code_sha256": code_hashes(*CODIGO),
        "catalogo": {k: {"familia": v[1], "descripcion": v[2]} for k, v in CATALOGO.items()},
        "ranking": orden,
        "resultados": {k: {x: y for x, y in v.items() if x != "origenes_detalle"}
                       for k, v in resultados.items()},
        "detalle_por_origen": {k: v["origenes_detalle"] for k, v in resultados.items()},
        "sensibilidad": sensible,
        "bootstrap_pareado": {
            "replicas": REPLICAS, "semilla": SEMILLA,
            "unidad_de_remuestreo": "orígenes, no días",
            "comparaciones": pruebas,
        },
        "plots": graficas,
        "completed_at_utc": ahora_utc(),
        "elapsed_seconds": round(time.perf_counter() - inicio, 1),
    }
    with destino.open("x", encoding="utf-8") as fichero:
        json.dump(informe, fichero, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Escrito {destino} y {len(graficas)} gráficas")
    return informe


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--solo", choices=["rapidos"], default=None,
                        help="omitir la familia clásica, que es la lenta")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.output, solo=args.solo)
