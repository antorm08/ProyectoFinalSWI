"""2. Pronóstico de la demanda diaria de pedidos (SARIMAX con estacionalidad semanal).

La demanda son **pedidos creados por día calendario, de cualquier estado**. La
ventana no es un literal: la decide `coverage_window` a partir de los huecos de
cobertura y de la cola truncada del archivo (ver `ml/demand_series.py`).

La especificación del modelo **no se elige aquí**. Se lee congelada del informe
que produce `ml/evaluate_demand.py`, para que el artefacto en producción sea
exactamente el que se midió. Sin ese informe, el script usa la especificación
por omisión y lo advierte.

Salidas:
  backend/app/models/demand_model.joblib   artefacto activo
  docs/demanda_metricas.json               protocolo, métricas y procedencia

Uso:
  python -m ml.train_demand --check-data    # auditoría de la serie, sin entrenar
  python -m ml.train_demand
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

if __package__:
    from .common import MODELS, ROOT
    from .demand_series import (
        PERIODO, ahora_utc, code_hashes, coverage, coverage_window, csv_hashes,
        daily_orders, final_holdout, mase, metrics, rel_mae, seasonal_naive, versiones,
    )
else:
    from common import MODELS, ROOT
    from demand_series import (
        PERIODO, ahora_utc, code_hashes, coverage, coverage_window, csv_hashes,
        daily_orders, final_holdout, mase, metrics, rel_mae, seasonal_naive, versiones,
    )

from app.services.demand_model import (
    ESQUEMA, Especificacion, ajustar, convergio, empaquetar, pronosticar, verificar,
)

DEFAULT_OUTPUT = MODELS / "demand_model.joblib"
DEFAULT_METRICS = ROOT / "docs" / "demanda_metricas.json"
DEFAULT_SELECTION = ROOT / "docs" / "resultados_demanda" / "metricas.json"
HOLDOUT = 28
NIVEL = 0.8

CODIGO = (
    "ml/train_demand.py",
    "ml/demand_series.py",
    "backend/app/services/demand_model.py",
    "backend/app/services/demand_calendar.py",
)


def leer_seleccion(ruta: Path) -> tuple[Especificacion, dict | None, dict]:
    """Especificación y calibración congeladas por el experimento.

    Sin informe se usa la especificación por omisión y **sin banda calibrada**,
    advirtiéndolo: una banda teórica sin medir es justo lo que este componente
    dejó de publicar.
    """
    if not Path(ruta).exists():
        spec = Especificacion()
        return spec, None, {
            "origen": "por omisión",
            "advertencia": (
                f"No se encontró {ruta}. Ejecuta `python -m ml.evaluate_demand` para "
                "elegir la especificación y calibrar la banda con evidencia."
            ),
            "nombre": spec.nombre,
        }
    informe = json.loads(Path(ruta).read_text(encoding="utf-8"))
    seleccion = informe.get("selection") or {}
    if not seleccion.get("spec"):
        raise SystemExit(f"{ruta} no contiene selection.spec; vuelve a lanzar la evaluación")
    spec = Especificacion.desde_dict(seleccion["spec"])
    calibracion = informe.get("calibration")
    if not calibracion:
        print("AVISO: el informe no trae calibración; la banda saldrá del intervalo teórico.")
    return spec, calibracion, {
        "origen": str(ruta.relative_to(ROOT)),
        "ganador": seleccion.get("ganador"),
        "criterio": seleccion.get("criterio"),
        "nombre": spec.nombre,
        "banda": "calibrada con los errores del backtesting" if calibracion else "teórica",
        "evaluacion_completada": informe.get("completed_at_utc"),
    }


def auditar(serie, ventana) -> dict:
    """Lo que `--check-data` imprime y el JSON conserva."""
    return {"serie_completa": dict(serie.attrs.get("audit", {})), **ventana.summary()}


def main(check_only: bool = False, output: Path = DEFAULT_OUTPUT,
         metrics_path: Path = DEFAULT_METRICS, selection: Path = DEFAULT_SELECTION) -> dict:
    inicio = time.perf_counter()
    arranque = ahora_utc()

    serie = daily_orders()
    ventana = coverage_window(serie)
    datos = ventana.serie
    auditoria = auditar(serie, ventana)

    print(f"Serie completa: {auditoria['serie_completa']['days_spanned']} días "
          f"({auditoria['serie_completa']['purchase_min'][:10]} a "
          f"{auditoria['serie_completa']['purchase_max'][:10]}), "
          f"{auditoria['serie_completa']['days_without_records']} sin registros")
    print(f"Ventana de cobertura: {auditoria['window']['inicio']} a {auditoria['window']['fin']} "
          f"({auditoria['window']['dias']} días, {auditoria['window']['pedidos']} pedidos)")
    heredada = auditoria["legacy_window"]
    print(f"Ventana heredada del script anterior: {heredada['inicio']} a {heredada['fin']} "
          f"({heredada['dias']} días, {heredada['pedidos']} pedidos)")
    print(f"Descartado: {auditoria['discarded']['pedidos_antes']} pedidos antes y "
          f"{auditoria['discarded']['pedidos_despues']} después de la ventana")
    if auditoria["first_truncated_day"]:
        print(f"Primer día con cobertura truncada: {auditoria['first_truncated_day']}")
    for hueco in auditoria["coverage_gaps"]:
        print(f"  hueco de {hueco['dias']} días: {hueco['inicio']} a {hueco['fin']}")

    if check_only:
        print("\n--check-data: no se entrena ni se escribe nada.")
        return auditoria

    spec, calibracion, procedencia = leer_seleccion(Path(selection))
    if "advertencia" in procedencia:
        print(f"AVISO: {procedencia['advertencia']}")
    print(f"Especificación ({procedencia['origen']}): {spec.nombre}")

    # Holdout final: se mide con el modelo que se va a servir, no con otro.
    train, test = final_holdout(datos, holdout=HOLDOUT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parcial = ajustar(train, spec)
    pronostico = pronosticar(parcial, train, spec, len(test), NIVEL, calibracion=calibracion)
    pronostico.index = test.index
    base = seasonal_naive(train, len(test))
    base.index = test.index
    evaluacion = {
        **metrics(test, pronostico["valor"]),
        "mase": mase(test, pronostico["valor"], train),
        "rel_mae": rel_mae(test, pronostico["valor"], train),
        "cobertura_80": coverage(test, pronostico["inferior"], pronostico["superior"]),
        "baseline_mae": metrics(test, base)["mae"],
        "periodo": f"{test.index[0].date()} a {test.index[-1].date()}",
        "convergio": convergio(parcial),
    }
    print(f"Holdout {evaluacion['periodo']}: MAE {evaluacion['mae']:.2f} pedidos/día | "
          f"MASE {evaluacion['mase']:.3f} | relMAE {evaluacion['rel_mae']:.3f} | "
          f"cobertura 80 % {evaluacion['cobertura_80']:.2f} "
          f"(baseline MAE {evaluacion['baseline_mae']:.2f})")
    if evaluacion["rel_mae"] >= 1:
        print("AVISO: el modelo NO bate a repetir la última semana en el holdout.")

    # Artefacto: reajuste con toda la ventana. Sus predicciones sobre el mismo
    # holdout no son evaluación fuera de muestra y no se publican como tal.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        completo = ajustar(datos, spec)
    if not convergio(completo):
        print("AVISO: la optimización del modelo final no declaró convergencia.")

    bundle = empaquetar(
        datos, spec, completo, calibracion=calibracion,
        window=auditoria["window"],
        metrics={k: v for k, v in evaluacion.items() if k != "convergio"},
        selection=procedencia,
        trained_at_utc=arranque,
        versions=versiones(),
        source_csv_sha256=csv_hashes(),
        anchor=auditoria["window"]["fin"],
    )
    comprobacion = verificar(bundle)
    print(f"Ida y vuelta del artefacto: desvío máximo {max(comprobacion['desvio_maximo'].values()):.2e}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    print(f"Guardado en {output} ({output.stat().st_size} bytes)")

    informe = {
        "started_at_utc": arranque,
        "protocol": {
            "target": "pedidos creados por día, cualquier estado",
            "ventana": "regla de cobertura, no literales de fecha",
            "holdout_dias": HOLDOUT,
            "nivel_banda": NIVEL,
            "nota": ("el artefacto se reajusta con toda la ventana; las métricas "
                     "publicadas son las del holdout, calculadas antes de ese reajuste"),
        },
        "versions": versiones(),
        "source_csv_sha256": csv_hashes(),
        "code_sha256": code_hashes(*CODIGO),
        "window": auditoria,
        "specification": {**spec.a_dict(), "nombre": spec.nombre, "procedencia": procedencia},
        "calibration": ({k: v for k, v in calibracion.items() if k != "errores_relativos"}
                        if calibracion else None),
        "holdout": evaluacion,
        "artifact": {
            "path": str(output.relative_to(ROOT)),
            "schema": ESQUEMA,
            "bytes": output.stat().st_size,
            "sha256": __import__("hashlib").sha256(output.read_bytes()).hexdigest(),
            "roundtrip": comprobacion,
            "convergio": convergio(completo),
        },
        "acceptance": {
            "objetivo_rel_mae": 1.0,
            "rel_mae": evaluacion["rel_mae"],
            "passed": bool(evaluacion["rel_mae"] < 1.0),
        },
        "completed_at_utc": ahora_utc(),
        "elapsed_seconds": round(time.perf_counter() - inicio, 1),
    }
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Métricas en {metrics_path}")
    return informe


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-data", action="store_true",
                        help="auditar la serie y la ventana sin entrenar ni escribir")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION,
                        help="informe de ml/evaluate_demand.py con la especificación congelada")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(check_only=args.check_data, output=args.output,
         metrics_path=args.metrics, selection=args.selection)
