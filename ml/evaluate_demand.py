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

from app.services.demand_model import Especificacion, ajustar, convergio, pronosticar

DEFAULT_OUTPUT = ROOT / "docs" / "resultados_demanda"
HORIZONTE = 28
N_ORIGENES = 8
PASO = 14
NIVELES = (0.8, 0.95)
BASELINE = "naive_estacional"

#: Candidatos que el backend sabe reconstruir y servir con banda. El artefacto
#: sale de aquí: un modelo sin intervalos no puede cumplir el contrato de la API.
SERVIBLES = ("sarima_heredado", "sarimax_log_calendario")

#: Criterio de aceptación, al estilo de TARGET_ACCURACY en ml/train_products.py.
OBJETIVO_REL_MAE = 1.0            # el modelo debe batir a repetir la última semana
OBJETIVO_COBERTURA_80 = (0.70, 0.90)

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


def elegir_orden(train: pd.Series, *, rapido: bool = False) -> tuple[Especificacion, dict]:
    """Rejilla pequeña por AIC, **solo sobre el primer origen de desarrollo**.

    El orden se congela aquí y se reutiliza en el resto de orígenes y en el
    holdout: así el protocolo no se optimiza contra sus propias particiones.
    """
    d, contraste = diferenciacion(train)
    pq = [(0, 0), (0, 1), (1, 0), (1, 1)] if rapido else [
        (p, q) for p in range(3) for q in range(3)
    ]
    estacionales = [(1, 0, 1)] if rapido else [
        (P, D, Q) for P in range(2) for D in range(2) for Q in range(2)
    ]
    intentos = []
    for p, q in pq:
        for P, D, Q in estacionales:
            spec = Especificacion(
                order=(p, d, q), seasonal_order=(P, D, Q, PERIODO), transform="log1p", exog=True
            )
            if p == q == P == Q == 0 and d == D == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    resultados = ajustar(train, spec)
                intentos.append({
                    "order": list(spec.order), "seasonal_order": list(spec.seasonal_order),
                    "aic": float(resultados.aic), "convergio": convergio(resultados),
                })
            except Exception as error:
                intentos.append({
                    "order": list(spec.order), "seasonal_order": list(spec.seasonal_order),
                    "error": str(error)[:120],
                })
    validos = [i for i in intentos if "aic" in i and np.isfinite(i["aic"])]
    if not validos:
        raise RuntimeError("Ningún orden de la rejilla se pudo ajustar")
    mejor = min(validos, key=lambda i: i["aic"])
    spec = Especificacion(
        order=tuple(mejor["order"]), seasonal_order=tuple(mejor["seasonal_order"]),
        transform="log1p", exog=True,
    )
    detalle = {
        "contraste_diferenciacion": contraste,
        "rejilla": sorted(intentos, key=lambda i: i.get("aic", float("inf"))),
        "elegido": {"order": mejor["order"], "seasonal_order": mejor["seasonal_order"],
                    "aic": mejor["aic"], "nombre": spec.nombre},
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

    spec, seleccion_orden = elegir_orden(particiones[0][0], rapido=rapido)
    print(f"Orden elegido por AIC: {spec.nombre} (AIC {seleccion_orden['elegido']['aic']:.1f})")

    candidatos = {
        BASELINE: candidato_naive,
        "media_dia_semana": candidato_dia_semana,
        "ets_semanal": candidato_ets,
        "sarima_heredado": candidato_sarimax(SPEC_HEREDADA),
        "sarimax_log_calendario": candidato_sarimax(spec),
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

    train_final, test_final = final_holdout(datos, holdout=horizonte)
    marco_final, extra_final = ejecutar_ganador(train_final, len(test_final), 0.8)
    marco_final.index = test_final.index
    base_final = seasonal_naive(train_final, len(test_final))
    base_final.index = test_final.index
    final = {
        **metrics(test_final, marco_final["valor"]),
        "mase": mase(test_final, marco_final["valor"], train_final),
        "rel_mae": rel_mae(test_final, marco_final["valor"], train_final),
        "periodo": f"{test_final.index[0].date()} a {test_final.index[-1].date()}",
        "baseline_mae": metrics(test_final, base_final)["mae"],
        **extra_final,
    }
    if "inferior" in marco_final:
        final["cobertura_80"] = coverage(test_final, marco_final["inferior"], marco_final["superior"])
    print(f"Holdout final ({final['periodo']}): MAE {final['mae']:.2f} "
          f"relMAE {final['rel_mae']:.3f}")

    spec_ganadora = spec if ganador == "sarimax_log_calendario" else (
        SPEC_HEREDADA if ganador == "sarima_heredado" else None
    )
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
