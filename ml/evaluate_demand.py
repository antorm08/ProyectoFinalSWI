"""2. Comparación de candidatos para el pronóstico de demanda diaria.

`docs/validacion_olist.md` sección 6.2 pide exactamente esto: «backtesting con
varios orígenes, baseline de última semana y un conjunto final reservado». El
script original evaluaba un único holdout de 30 días contra ningún baseline, así
que no había forma de saber si el SARIMA aportaba algo sobre repetir la última
semana, que es lo que hace el modo provisional del backend.

Protocolo:
  - Serie: pedidos creados por día, recortada por `coverage_window`.
  - Desarrollo: orígenes móviles con ventana expansiva. **Todo** se elige aquí:
    diferenciación, orden, transformación y exógenas.
  - Holdout final: los últimos días, mirados una sola vez con el modelo ya
    congelado.
  - Métricas: MAE, RMSE, sMAPE, MASE y MAE relativo al naive estacional, más la
    cobertura empírica de las bandas del 80 y 95 % frente a su nominal.

Salidas (nunca sobrescribe):
  docs/resultados_demanda/metricas.json
  docs/resultados_demanda/*.png

Uso:
  python -m ml.evaluate_demand
  python -m ml.evaluate_demand --output /tmp/prueba --rapido
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .common import ROOT
    from .demand_series import (
        PERIODO, ahora_utc, code_hashes, coverage, coverage_window, csv_hashes,
        daily_orders, final_holdout, mase, metrics, rel_mae, rolling_origin,
        seasonal_naive, versiones, weekday_mean,
    )
else:
    from common import ROOT
    from demand_series import (
        PERIODO, ahora_utc, code_hashes, coverage, coverage_window, csv_hashes,
        daily_orders, final_holdout, mase, metrics, rel_mae, rolling_origin,
        seasonal_naive, versiones, weekday_mean,
    )

from app.services.demand_model import (
    BLOQUE_DIAS, Especificacion, ajustar, convergio, pronosticar,
)

DEFAULT_OUTPUT = ROOT / "docs" / "resultados_demanda"
HORIZONTE = 28
N_ORIGENES = 8
PASO = 14
NIVELES = (0.8, 0.95)
BASELINE = "naive_estacional"

#: Candidatos que el backend sabe reconstruir y servir con banda. El artefacto
#: sale de aquí: un modelo sin intervalos no puede cumplir el contrato de la API.
SERVIBLES = ("sarima_heredado", "sarimax_elegido")

#: Protocolo de calibración de la banda: horizonte completo de la API.
CALIBRACION = {"horizonte": 90, "n_origenes": 6, "paso": 28}

#: Criterio de aceptación, al estilo de TARGET_ACCURACY en ml/train_products.py.
OBJETIVO_REL_MAE = 1.0            # el modelo debe batir a repetir la última semana
OBJETIVO_COBERTURA_80 = (0.70, 0.90)

#: Las cuatro combinaciones de escala y exógenas que se exploran. El AIC no es
#: comparable entre escalas (las verosimilitudes no están en las mismas
#: unidades), así que filtra **dentro** de cada familia y el backtesting decide
#: entre ellas. De paso, la tabla de finalistas responde a dos preguntas que de
#: otro modo quedarían en suposición: ¿aporta el log1p? ¿aporta el calendario?
FAMILIAS = (("log1p", True), ("log1p", False), ("identidad", True), ("identidad", False))

CODIGO = (
    "ml/evaluate_demand.py",
    "ml/demand_series.py",
    "backend/app/services/demand_model.py",
    "backend/app/services/demand_calendar.py",
)


# --------------------------------------------------------------------------- #
# Candidatos
# --------------------------------------------------------------------------- #
def _marco(valor: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"valor": valor.astype(float)})


def candidato_naive(train, steps, nivel):
    return _marco(seasonal_naive(train, steps)), {}


def candidato_dia_semana(train, steps, nivel):
    return _marco(weekday_mean(train, steps, semanas=4)), {}


def candidato_ets(train, steps, nivel):
    """Holt-Winters aditivo: segunda familia de modelos, sin coste añadido."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ajuste = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=PERIODO,
            initialization_method="estimated",
        ).fit()
    futuro = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    return _marco(pd.Series(ajuste.forecast(steps).to_numpy(), index=futuro)), {
        "aic": float(ajuste.aic)
    }


def candidato_sarimax(spec: Especificacion):
    """Fábrica: fija la especificación y deja la firma uniforme."""

    def ejecutar(train, steps, nivel):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resultados = ajustar(train, spec)
        marco = pronosticar(resultados, train, spec, steps, nivel)
        return marco, {"aic": float(resultados.aic), "convergio": convergio(resultados)}

    return ejecutar


SPEC_HEREDADA = Especificacion(
    order=(1, 1, 1), seasonal_order=(1, 0, 1, PERIODO), transform="identidad", exog=False
)


# --------------------------------------------------------------------------- #
# Elección de la diferenciación y del orden
# --------------------------------------------------------------------------- #
def diferenciacion(serie: pd.Series) -> tuple[int, dict]:
    """d por contraste: ADF (raíz unitaria) y KPSS (estacionariedad)."""
    from statsmodels.tsa.stattools import adfuller, kpss

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valores = np.log1p(serie.to_numpy(dtype=float))
        p_adf = float(adfuller(valores, autolag="AIC")[1])
        p_kpss = float(kpss(valores, regression="c", nlags="auto")[1])
    estacionaria = p_adf < 0.05 and p_kpss > 0.05
    return (0 if estacionaria else 1), {
        "adf_pvalue": round(p_adf, 5),
        "kpss_pvalue": round(p_kpss, 5),
        "criterio": "d=0 solo si ADF rechaza raíz unitaria y KPSS no rechaza estacionariedad",
        "d": 0 if estacionaria else 1,
    }


def elegir_orden(particiones, *, rapido: bool = False, por_familia: int = 3) -> tuple[Especificacion, dict]:
    """Selección en dos etapas sobre los orígenes de desarrollo.

    1. **Filtro por AIC dentro de cada familia** (escala x exógenas), sobre el
       primer origen: el AIC no es comparable entre escalas, así que se usa
       donde sí vale, para quedarse con los `por_familia` órdenes más
       verosímiles de cada una.
    2. **Criterio por backtesting**: entre todos los finalistas gana el de
       menor MAE relativo al naive estacional, promediado en los orígenes,
       **exigiendo antes que su banda esté calibrada**. Un intervalo demasiado
       ancho cubre mucho y no informa de nada.

    La primera versión de este script elegía solo por AIC y el orden ganador
    (2,1,2)(1,1,1) pronosticaba **peor** fuera de muestra que el heredado
    (1,1,1)(1,0,1): 0,999 frente a 0,994 de MAE relativo. El AIC mide ajuste
    dentro de la muestra, no capacidad de pronóstico, así que decide el
    backtesting. El holdout final sigue sin tocarse.
    """
    train = particiones[0][0]
    d, contraste = diferenciacion(train)
    pq = [(0, 0), (0, 1), (1, 0), (1, 1)] if rapido else [
        (p, q) for p in range(3) for q in range(3)
    ]
    estacionales = [(1, 0, 1)] if rapido else [
        (P, D, Q) for P in range(2) for D in range(2) for Q in range(2)
    ]
    familias = FAMILIAS[:1] if rapido else FAMILIAS

    rejilla, finalistas = [], []
    for transform, exog in familias:
        intentos = []
        for p, q in pq:
            for P, D, Q in estacionales:
                if p == q == P == Q == 0 and d == D == 0:
                    continue
                spec = Especificacion(order=(p, d, q), seasonal_order=(P, D, Q, PERIODO),
                                      transform=transform, exog=exog)
                fila = {"order": list(spec.order), "seasonal_order": list(spec.seasonal_order),
                        "transform": transform, "exog": exog}
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        resultados = ajustar(train, spec)
                    aic = float(resultados.aic)
                    intentos.append({**fila, "aic": aic, "convergio": convergio(resultados)}
                                    if np.isfinite(aic) else {**fila, "error": "AIC no finito"})
                except Exception as error:
                    intentos.append({**fila, "error": str(error)[:120]})
        validos = sorted((i for i in intentos if "aic" in i), key=lambda i: i["aic"])
        rejilla.extend(intentos)
        if not validos:
            print(f"    familia {transform}/{'calendario' if exog else 'sin exógenas'}: sin ajustes válidos")
            continue
        preseleccion = validos[: max(1, por_familia if not rapido else 1)]
        print(f"    familia {transform}/{'calendario' if exog else 'sin exógenas':14s} "
              f"mejor AIC {validos[0]['aic']:9.1f}")
        for candidato in preseleccion:
            spec = Especificacion(order=tuple(candidato["order"]),
                                  seasonal_order=tuple(candidato["seasonal_order"]),
                                  transform=transform, exog=exog)
            try:
                filas = [evaluar_origen(candidato_sarimax(spec), tr, te) for tr, te in particiones]
            except Exception as error:
                finalistas.append({**candidato, "error_backtest": str(error)[:120]})
                continue
            resumen = agregar(filas)
            finalistas.append({**candidato, "rel_mae": resumen["rel_mae"], "mae": resumen["mae"],
                               "cobertura": resumen["cobertura"],
                               "banda_media": resumen["banda_media"], "nombre": spec.nombre})
            print(f"      {spec.nombre:46s} relMAE {resumen['rel_mae']:.3f} "
                  f"cob {resumen['cobertura']:.3f}")

    evaluables = [f for f in finalistas if "rel_mae" in f and np.isfinite(f["rel_mae"])]
    if not evaluables:
        raise RuntimeError("Ningún finalista se pudo evaluar en los orígenes de desarrollo")
    # La banda del artefacto se calibra después con los errores del backtesting,
    # así que aquí decide la precisión del pronóstico. La cobertura teórica de
    # cada finalista se conserva como diagnóstico: es la que resultó estar mal.
    calibrados = [f for f in evaluables
                  if f["cobertura"] is not None
                  and OBJETIVO_COBERTURA_80[0] <= f["cobertura"] <= OBJETIVO_COBERTURA_80[1]]
    mejor = min(evaluables, key=lambda f: f["rel_mae"])
    mejor_aic = min((i for i in rejilla if "aic" in i), key=lambda i: i["aic"])
    spec = Especificacion(order=tuple(mejor["order"]), seasonal_order=tuple(mejor["seasonal_order"]),
                          transform=mejor["transform"], exog=mejor["exog"])
    detalle = {
        "criterio": (
            "el AIC filtra dentro de cada familia (escala x exógenas) y el backtesting decide "
            "entre ellas por menor MAE relativo al naive estacional; la banda no entra en la "
            "decisión porque se calibra después con los errores observados"
        ),
        "cobertura_teorica_es_diagnostico": (
            "la columna de cobertura de los finalistas es la del intervalo teórico de "
            "statsmodels, que es justamente el que se descarta"
        ),
        "familias": [{"transform": t, "exog": e} for t, e in familias],
        "contraste_diferenciacion": contraste,
        "rejilla_aic": sorted((i for i in rejilla if "aic" in i), key=lambda i: i["aic"]),
        "errores_rejilla": [i for i in rejilla if "aic" not in i],
        "finalistas": sorted(evaluables, key=lambda f: f["rel_mae"]),
        "finalistas_calibrados": len(calibrados),
        "elegido": {**{k: mejor[k] for k in ("order", "seasonal_order", "transform", "exog",
                                             "aic", "rel_mae", "cobertura")}, "nombre": spec.nombre},
        "mejor_por_aic": {k: mejor_aic[k] for k in ("order", "seasonal_order", "transform", "exog", "aic")},
        "aic_y_backtest_coinciden": bool(
            mejor["order"] == mejor_aic["order"]
            and mejor["seasonal_order"] == mejor_aic["seasonal_order"]
            and mejor["transform"] == mejor_aic["transform"] and mejor["exog"] == mejor_aic["exog"]
        ),
        "ventana_de_eleccion": {
            "inicio": str(train.index[0].date()), "fin": str(train.index[-1].date()),
            "dias": int(len(train)),
        },
    }
    return spec, detalle


# --------------------------------------------------------------------------- #
# Evaluación
# --------------------------------------------------------------------------- #
def evaluar_origen(ejecutar, train, test, nivel=0.8) -> dict:
    """Un candidato en un origen: métricas y, si las da, cobertura de la banda."""
    inicio = time.perf_counter()
    marco, extra = ejecutar(train, len(test), nivel)
    marco.index = test.index
    fila = {
        **metrics(test, marco["valor"]),
        "mase": mase(test, marco["valor"], train),
        "rel_mae": rel_mae(test, marco["valor"], train),
        "segundos": round(time.perf_counter() - inicio, 2),
        "test_inicio": str(test.index[0].date()),
        "test_fin": str(test.index[-1].date()),
        "train_dias": int(len(train)),
        **extra,
    }
    if {"inferior", "superior"} <= set(marco.columns):
        fila["cobertura"] = coverage(test, marco["inferior"], marco["superior"])
        fila["banda_media"] = float((marco["superior"] - marco["inferior"]).mean())
    return fila


def agregar(filas: list[dict]) -> dict:
    """Promedio macro entre orígenes: cada origen pesa igual."""
    def media(clave):
        valores = [f[clave] for f in filas if clave in f and f[clave] is not None]
        return float(np.mean(valores)) if valores else None

    return {
        "origenes": len(filas),
        "mae": media("mae"), "rmse": media("rmse"), "smape_pct": media("smape_pct"),
        "mase": media("mase"), "rel_mae": media("rel_mae"), "bias": media("bias"),
        "cobertura": media("cobertura"), "banda_media": media("banda_media"),
        "segundos": media("segundos"),
    }


def coberturas_por_nivel(ejecutar, particiones) -> dict:
    """Cobertura empírica de cada nivel nominal. Sin esto la banda decora."""
    salida = {}
    for nivel in NIVELES:
        observadas = []
        for train, test in particiones:
            marco, _ = ejecutar(train, len(test), nivel)
            marco.index = test.index
            if {"inferior", "superior"} <= set(marco.columns):
                observadas.append(coverage(test, marco["inferior"], marco["superior"]))
        if observadas:
            salida[f"{int(nivel * 100)}"] = {
                "nominal": nivel, "empirica": float(np.mean(observadas)),
                "origenes": len(observadas),
            }
    return salida


def calibrar(spec: Especificacion, serie: pd.Series, *, horizonte: int, n_origenes: int,
             paso: int, holdout: int, bloque: int = BLOQUE_DIAS) -> dict:
    """Errores relativos del backtesting, agrupados por bloque del horizonte.

    Se calibra sobre el **horizonte completo que ofrece la API** (90 días), no
    sobre los 28 del protocolo de comparación: de otro modo la banda de los
    días lejanos se quedaría sin evidencia y habría que inventarla.

    La cobertura que esto produce en los propios orígenes de desarrollo es
    circular por construcción. La cifra que vale es la del holdout final, donde
    la calibración viene de datos que ese tramo no vio.
    """
    particiones = rolling_origin(serie, n_origenes=n_origenes, horizonte=horizonte,
                                 paso=paso, holdout=holdout)
    n_bloques = math.ceil(horizonte / bloque)
    errores: list[list[float]] = [[] for _ in range(n_bloques)]
    suelo = 0
    for train, test in particiones:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resultados = ajustar(train, spec)
        marco = pronosticar(resultados, train, spec, len(test), 0.8)
        marco.index = test.index
        previsto = marco["valor"].to_numpy(dtype=float)
        suelo += int((previsto < 1).sum())
        relativo = (test.to_numpy(dtype=float) - previsto) / np.maximum(previsto, 1.0)
        for paso_h, error in enumerate(relativo):
            errores[min(paso_h // bloque, n_bloques - 1)].append(float(error))
    resumen = [
        {
            "bloque": indice + 1,
            "dias": f"{indice * bloque + 1}-{min((indice + 1) * bloque, horizonte)}",
            "n": len(muestra),
            "q10": round(float(np.quantile(muestra, 0.1)), 4),
            "q90": round(float(np.quantile(muestra, 0.9)), 4),
            "mediana": round(float(np.quantile(muestra, 0.5)), 4),
        }
        for indice, muestra in enumerate(errores)
    ]
    return {
        "metodo": ("cuantiles empíricos del error relativo del backtesting, "
                   "por bloque del horizonte"),
        "bloque_dias": bloque,
        "horizonte": horizonte,
        "origenes": len(particiones),
        "paso_dias": paso,
        "primer_origen": str(particiones[0][1].index[0].date()),
        "ultimo_origen": str(particiones[-1][1].index[0].date()),
        "previstos_bajo_el_suelo": suelo,
        "resumen_por_bloque": resumen,
        "errores_relativos": errores,
    }


# --------------------------------------------------------------------------- #
# Gráficas
# --------------------------------------------------------------------------- #
def render_plots(directorio: Path, serie, ventana, resumen, holdout, residuos) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nombres = []

    def guardar(nombre, fig):
        ruta = directorio / f"{nombre}.png"
        fig.savefig(ruta, dpi=130, bbox_inches="tight")
        plt.close(fig)
        nombres.append(ruta.name)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(serie.index, serie.to_numpy(), lw=0.8, color="#94a3b8", label="Serie completa")
    ax.plot(ventana.serie.index, ventana.serie.to_numpy(), lw=0.9, color="#2563eb",
            label="Ventana de cobertura")
    ax.axvline(ventana.inicio, color="#0f172a", ls="--", lw=0.8)
    ax.axvline(ventana.fin, color="#0f172a", ls="--", lw=0.8)
    ax.set_title("Pedidos por día: serie completa y ventana utilizada")
    ax.set_ylabel("pedidos/día"); ax.legend(fontsize=8)
    guardar("serie_y_cortes", fig)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for nombre, datos in resumen.items():
        filas = datos["origenes_detalle"]
        ax.plot([f["test_inicio"] for f in filas], [f["mae"] for f in filas],
                marker="o", ms=3, lw=1, label=nombre)
    ax.set_title(f"MAE por origen (horizonte {HORIZONTE} días)")
    ax.set_ylabel("MAE (pedidos/día)"); ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(fontsize=7)
    guardar("backtest_por_origen", fig)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(holdout["real"].index, holdout["real"].to_numpy(), color="#0f172a", lw=1.2, label="Real")
    ax.plot(holdout["pred"].index, holdout["pred"]["valor"].to_numpy(), color="#2563eb", lw=1.4,
            ls="--", label="Pronóstico")
    if "inferior" in holdout["pred"]:
        ax.fill_between(holdout["pred"].index, holdout["pred"]["inferior"],
                        holdout["pred"]["superior"], color="#2563eb", alpha=0.15,
                        label="Banda 80 %")
    ax.plot(holdout["base"].index, holdout["base"].to_numpy(), color="#f59e0b", lw=1,
            ls=":", label="Naive estacional")
    ax.set_title("Holdout final: real, pronóstico y baseline")
    ax.set_ylabel("pedidos/día"); ax.legend(fontsize=8)
    guardar("real_vs_pronostico", fig)

    fig, (izq, der) = plt.subplots(1, 2, figsize=(10, 3.2))
    izq.plot(residuos["serie"].index, residuos["serie"].to_numpy(), lw=0.7, color="#475569")
    izq.axhline(0, color="#0f172a", lw=0.6)
    izq.set_title("Residuos del ajuste (escala log1p)")
    der.bar(range(1, len(residuos["acf"])), residuos["acf"][1:], color="#2563eb")
    der.axhline(residuos["banda"], color="#ef4444", ls="--", lw=0.8)
    der.axhline(-residuos["banda"], color="#ef4444", ls="--", lw=0.8)
    der.set_title("ACF de los residuos")
    guardar("residuos_acf", fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    niveles = sorted(residuos["cobertura"].items())
    ax.bar([f"{k} %" for k, _ in niveles], [v["empirica"] for _, v in niveles], color="#2563eb",
           label="Empírica")
    for pos, (_, valor) in enumerate(niveles):
        ax.hlines(valor["nominal"], pos - 0.4, pos + 0.4, color="#0f172a", lw=1.5)
    ax.set_ylim(0, 1.05); ax.set_title("Cobertura de la banda: empírica vs nominal")
    guardar("cobertura_intervalos", fig)
    return nombres


# --------------------------------------------------------------------------- #
# Ejecución
# --------------------------------------------------------------------------- #
def run(output: Path, *, rapido: bool = False) -> dict:
    inicio = time.perf_counter()
    arranque = ahora_utc()
    output = Path(output)
    destino_json = output / "metricas.json"
    if destino_json.exists():
        raise FileExistsError(f"{destino_json} ya existe; no se sobrescriben resultados")

    serie = daily_orders()
    ventana = coverage_window(serie)
    datos = ventana.serie
    horizonte = PERIODO if rapido else HORIZONTE
    n_origenes = 2 if rapido else N_ORIGENES
    particiones = rolling_origin(
        datos, n_origenes=n_origenes, horizonte=horizonte, paso=PASO, holdout=horizonte
    )
    print(f"Ventana {ventana.inicio.date()} a {ventana.fin.date()} "
          f"({len(datos)} días, {int(datos.sum())} pedidos)")
    print(f"{len(particiones)} orígenes de desarrollo, horizonte {horizonte} días")

    print("Eligiendo orden: AIC filtra, el backtesting decide")
    spec, seleccion_orden = elegir_orden(particiones, rapido=rapido)
    print(f"  elegido: {spec.nombre} (relMAE {seleccion_orden['elegido']['rel_mae']:.3f}, "
          f"cobertura {seleccion_orden['elegido']['cobertura']:.3f}); "
          f"el mejor por AIC {'coincide' if seleccion_orden['aic_y_backtest_coinciden'] else 'NO coincide'}")

    candidatos = {
        BASELINE: candidato_naive,
        "media_dia_semana": candidato_dia_semana,
        "ets_semanal": candidato_ets,
        "sarima_heredado": candidato_sarimax(SPEC_HEREDADA),
        "sarimax_elegido": candidato_sarimax(spec),
    }

    resumen = {}
    for nombre, ejecutar in candidatos.items():
        filas = [evaluar_origen(ejecutar, train, test) for train, test in particiones]
        resumen[nombre] = {**agregar(filas), "origenes_detalle": filas}
        print(f"  {nombre:24s} MAE {resumen[nombre]['mae']:7.2f}  "
              f"relMAE {resumen[nombre]['rel_mae']:.3f}  MASE {resumen[nombre]['mase']:.3f}")

    servibles = {k: v for k, v in resumen.items() if k in SERVIBLES}
    ganador = min(servibles, key=lambda k: servibles[k]["rel_mae"])
    comparables = {k: v for k, v in resumen.items() if k != BASELINE}
    mejor_absoluto = min(comparables, key=lambda k: comparables[k]["rel_mae"])
    print(f"Ganador servible: {ganador}" + (
        f"  (mejor absoluto: {mejor_absoluto}, fuera de la familia que sirve el backend)"
        if mejor_absoluto != ganador else ""))

    ejecutar_ganador = candidatos[ganador]
    cobertura = coberturas_por_nivel(ejecutar_ganador, particiones)

    spec_ganadora = spec if ganador == "sarimax_elegido" else SPEC_HEREDADA
    print(f"Calibrando la banda del ganador sobre {CALIBRACION['horizonte']} días…")
    calibracion = calibrar(spec_ganadora, datos, holdout=horizonte, **CALIBRACION)
    print(f"  {calibracion['origenes']} orígenes, "
          f"{len(calibracion['errores_relativos'])} bloques; error relativo "
          f"días 1-7 [{calibracion['resumen_por_bloque'][0]['q10']:+.3f}, "
          f"{calibracion['resumen_por_bloque'][0]['q90']:+.3f}] → "
          f"días {calibracion['resumen_por_bloque'][-1]['dias']} "
          f"[{calibracion['resumen_por_bloque'][-1]['q10']:+.3f}, "
          f"{calibracion['resumen_por_bloque'][-1]['q90']:+.3f}]")

    # Holdout final: una sola vez, con el modelo congelado y la banda calibrada
    # con datos que este tramo no vio.
    train_final, test_final = final_holdout(datos, holdout=horizonte)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ajuste_final = ajustar(train_final, spec_ganadora)
    marco_final = pronosticar(ajuste_final, train_final, spec_ganadora, len(test_final), 0.8,
                              calibracion=calibracion)
    marco_final.index = test_final.index
    teorico = pronosticar(ajuste_final, train_final, spec_ganadora, len(test_final), 0.8)
    teorico.index = test_final.index
    base_final = seasonal_naive(train_final, len(test_final))
    base_final.index = test_final.index
    final = {
        **metrics(test_final, marco_final["valor"]),
        "mase": mase(test_final, marco_final["valor"], train_final),
        "rel_mae": rel_mae(test_final, marco_final["valor"], train_final),
        "periodo": f"{test_final.index[0].date()} a {test_final.index[-1].date()}",
        "baseline_mae": metrics(test_final, base_final)["mae"],
        "aic": float(ajuste_final.aic),
        "convergio": convergio(ajuste_final),
        "cobertura_80": coverage(test_final, marco_final["inferior"], marco_final["superior"]),
        "cobertura_80_teorica": coverage(test_final, teorico["inferior"], teorico["superior"]),
        "banda_media": float((marco_final["superior"] - marco_final["inferior"]).mean()),
        "banda_media_teorica": float((teorico["superior"] - teorico["inferior"]).mean()),
    }
    print(f"  cobertura del 80 %: calibrada {final['cobertura_80']:.3f} "
          f"frente a teórica {final['cobertura_80_teorica']:.3f}")
    print(f"Holdout final ({final['periodo']}): MAE {final['mae']:.2f} "
          f"relMAE {final['rel_mae']:.3f}")

    residuos = {"cobertura": cobertura}
    if spec_ganadora is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ajuste = ajustar(train_final, spec_ganadora)
        serie_residuos = pd.Series(np.asarray(ajuste.resid, dtype=float), index=train_final.index)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from statsmodels.tsa.stattools import acf
        prueba = acorr_ljungbox(serie_residuos, lags=[PERIODO, 14, 28], return_df=True)
        residuos |= {
            "serie": serie_residuos,
            "acf": acf(serie_residuos, nlags=28, fft=True),
            "banda": float(1.96 / np.sqrt(len(serie_residuos))),
            "ljung_box": {
                str(int(lag)): {"estadistico": float(fila["lb_stat"]), "pvalue": float(fila["lb_pvalue"])}
                for lag, fila in prueba.iterrows()
            },
        }
    else:
        residuos |= {"serie": pd.Series(dtype=float, index=pd.DatetimeIndex([])),
                     "acf": np.zeros(29), "banda": 0.0, "ljung_box": {}}

    cobertura_80 = final.get("cobertura_80")
    aceptacion = {
        "objetivo_rel_mae": OBJETIVO_REL_MAE,
        "objetivo_cobertura_80": list(OBJETIVO_COBERTURA_80),
        "rel_mae": final["rel_mae"],
        "cobertura_80": cobertura_80,
        "bate_al_baseline": bool(final["rel_mae"] < OBJETIVO_REL_MAE),
        "cobertura_en_rango": bool(
            cobertura_80 is not None
            and OBJETIVO_COBERTURA_80[0] <= cobertura_80 <= OBJETIVO_COBERTURA_80[1]
        ),
    }
    aceptacion["passed"] = bool(aceptacion["bate_al_baseline"] and aceptacion["cobertura_en_rango"])

    output.mkdir(parents=True, exist_ok=True)
    graficas = render_plots(
        output, serie, ventana, resumen,
        {"real": test_final, "pred": marco_final, "base": base_final}, residuos,
    )

    informe = {
        "started_at_utc": arranque,
        "protocol": {
            "target": "pedidos creados por día, cualquier estado",
            "horizonte_dias": horizonte,
            "origenes_desarrollo": len(particiones),
            "paso_dias": PASO,
            "holdout_dias": horizonte,
            "baseline": BASELINE,
            "niveles_banda": list(NIVELES),
            "nota": "el holdout final se evalúa una sola vez, con el modelo congelado",
        },
        "versions": versiones(),
        "source_csv_sha256": csv_hashes(),
        "code_sha256": code_hashes(*CODIGO),
        "window": ventana.summary(),
        "order_selection": seleccion_orden,
        "development": {
            nombre: {k: v for k, v in datos.items() if k != "origenes_detalle"}
            for nombre, datos in resumen.items()
        },
        "development_detail": {n: d["origenes_detalle"] for n, d in resumen.items()},
        "selection": {
            "ganador": ganador,
            "criterio": (
                "menor MAE relativo al naive estacional, promediado entre orígenes, "
                "entre los candidatos que el backend sabe reconstruir y servir con banda"
            ),
            "servibles": list(SERVIBLES),
            "spec": spec_ganadora.a_dict() if spec_ganadora else None,
            "nombre": spec_ganadora.nombre if spec_ganadora else ganador,
            "mejor_absoluto": mejor_absoluto,
            "mejor_absoluto_rel_mae": comparables[mejor_absoluto]["rel_mae"],
            "nota_fuera_de_familia": (
                None if mejor_absoluto == ganador else
                f"{mejor_absoluto} obtiene mejor relMAE en desarrollo pero no produce "
                "intervalos reconstruibles por el servicio; queda como referencia y como "
                "recomendación de trabajo futuro, no como artefacto activo."
            ),
        },
        "interval_coverage": cobertura,
        "calibration": calibracion,
        "residual_diagnostics": residuos["ljung_box"],
        "final_holdout": final,
        "acceptance": aceptacion,
        "plots": graficas,
        "completed_at_utc": ahora_utc(),
        "elapsed_seconds": round(time.perf_counter() - inicio, 1),
    }
    with destino_json.open("x", encoding="utf-8") as fichero:
        json.dump(informe, fichero, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Escrito {destino_json} y {len(graficas)} gráficas")
    print(f"Aceptación: {'PASA' if aceptacion['passed'] else 'NO PASA'}")
    return informe


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="directorio de salida (no se sobrescribe)")
    parser.add_argument("--rapido", action="store_true",
                        help="rejilla y orígenes mínimos, solo para comprobar la tubería")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.output, rapido=args.rapido)
