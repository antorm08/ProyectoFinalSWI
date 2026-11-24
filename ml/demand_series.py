"""Serie diaria de demanda, ventana por cobertura, baselines y backtesting.

Importar este módulo no lee CSV, no entrena y no escribe artefactos.

La demanda se define como **pedidos creados por día calendario, de cualquier
estado**, a partir de `order_purchase_timestamp`. Filtrar por `delivered`
introduciría sesgo de supervivencia: un pedido cancelado también fue demanda.

Sustituye los dos literales del script original (`2017-01-01` y `2018-08-15`)
por una regla medible, porque `docs/validacion_olist.md` sección 5.2 mide que la
serie completa tiene 140 días sin registros y que septiembre y octubre de 2018
solo contienen 16 y 4 compras: pronosticar sobre esa cola produce una caída
artificial, y tratar los ceros de enero de 2017 como demanda nula es falso.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .common import OLIST, ROOT, load, require_unique
else:  # ejecución como script desde ml/
    from common import OLIST, ROOT, load, require_unique

# El calendario vive en el backend y se comparte con el servicio, igual que
# ml/evaluate_products.py importa app.services.product_service.
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))
from app.services.demand_calendar import COLUMNAS as EXOG_COLUMNAS  # noqa: E402
from app.services.demand_calendar import calendar_exog  # noqa: E402

#: Ventana del script original, conservada solo para reproducir su diagnóstico.
VENTANA_HEREDADA = ("2017-01-01", "2018-08-15")

#: Estacionalidad de la serie: semanal.
PERIODO = 7

__all__ = [
    "EXOG_COLUMNAS", "PERIODO", "VENTANA_HEREDADA", "Ventana", "ahora_utc",
    "calendar_exog", "code_hashes", "coverage", "coverage_window", "csv_hashes",
    "daily_orders", "final_holdout", "mase", "metrics", "rel_mae",
    "rolling_origin", "seasonal_naive", "sha256", "versiones", "weekday_mean",
]


# --------------------------------------------------------------------------- #
# Serie
# --------------------------------------------------------------------------- #
def daily_orders(source: pd.DataFrame | None = None) -> pd.Series:
    """Pedidos creados por día, reindexada a frecuencia diaria completa.

    Los días sin ningún registro quedan a cero; distinguir esos ceros de una
    demanda realmente nula es trabajo de `coverage_window`, no de aquí.
    """
    orders = load("orders") if source is None else source
    require_unique(orders, ["order_id"], "Pedidos")
    marcas = pd.to_datetime(orders["order_purchase_timestamp"], errors="raise")
    if marcas.isna().any():
        raise ValueError("Un pedido no tiene fecha de compra")
    conteo = marcas.dt.floor("D").value_counts().sort_index()
    serie = conteo.asfreq("D", fill_value=0).astype(float)
    serie.name = "pedidos"
    serie.index.name = "fecha"
    sin_registros = int((serie == 0).sum())
    serie.attrs["audit"] = {
        "source_orders": int(len(orders)),
        "purchase_min": str(marcas.min()),
        "purchase_max": str(marcas.max()),
        "days_spanned": int(len(serie)),
        "days_without_records": sin_registros,
        "orders_in_series": int(serie.sum()),
    }
    return serie


# --------------------------------------------------------------------------- #
# Ventana de cobertura
# --------------------------------------------------------------------------- #
@dataclass
class Ventana:
    """Tramo de cobertura fiable y la auditoría de por qué es ese y no otro."""
    serie: pd.Series
    inicio: pd.Timestamp
    fin: pd.Timestamp
    audit: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return dict(self.audit)


def _rachas(mask: pd.Series) -> list[tuple[int, int]]:
    """Rachas contiguas de True como pares [inicio, fin] de posiciones."""
    valores = mask.to_numpy()
    if not valores.any():
        return []
    cambios = np.flatnonzero(np.diff(valores.astype(np.int8)))
    bordes = np.concatenate(([0], cambios + 1, [len(valores)]))
    rachas = []
    for desde, hasta in zip(bordes[:-1], bordes[1:]):
        if valores[desde]:
            rachas.append((int(desde), int(hasta) - 1))
    return rachas


def coverage_window(
    serie: pd.Series,
    *,
    hueco_minimo: int = 2,
    fraccion: float = 0.25,
    referencia: int = 180,
) -> Ventana:
    """Recorta la serie a su tramo contiguo más largo sin huecos ni cola truncada.

    Dos defectos distintos, dos reglas distintas:

    - **Hueco de cobertura**: `hueco_minimo` o más días consecutivos sin ningún
      pedido. Un día sin registros rodeado de actividad no es demanda nula.
    - **Cola truncada**: la suma móvil de 7 días cae por debajo de `fraccion`
      del nivel habitual, medido como la mediana de las sumas móviles de los
      `referencia` días anteriores (desplazada una semana para no mirarse a sí
      misma). Solo puede dispararse a la baja, así que la rampa de crecimiento
      de 2017 no la activa.

    El agotamiento del archivo es **gradual**, no un corte limpio: en Olist los
    conteos caen durante una semana larga antes de llegar a cero. Un umbral
    absoluto detecta el final del descenso, no su comienzo, y dejaría dentro
    días ya infrarregistrados que un modelo interpretaría como caída de la
    demanda. Por eso, tras localizar el corte, se retrocede mientras la media
    móvil de 7 días siga bajando: se descarta el descenso final completo. La
    marcha atrás está anclada al final de la serie, así que no puede recortar
    un valle interior.

    Se devuelve la racha válida más larga; los recortes y sus motivos quedan en
    la auditoría.
    """
    if not isinstance(serie.index, pd.DatetimeIndex):
        raise TypeError("coverage_window necesita una serie con índice de fechas")
    if serie.index.freq is None:
        raise ValueError("La serie debe estar reindexada a frecuencia diaria")
    if len(serie) < PERIODO * 2:
        raise ValueError("Serie demasiado corta para evaluar su cobertura")

    huecos = [r for r in _rachas(serie.eq(0)) if r[1] - r[0] + 1 >= hueco_minimo]
    suma7 = serie.rolling(PERIODO).sum()
    nivel = suma7.rolling(referencia).median().shift(PERIODO)
    truncado = (suma7 < fraccion * nivel).fillna(False)

    descenso_desde = None
    if truncado.any():
        posicion = int(serie.index.get_loc(truncado[truncado].index[0]))
        valores = suma7.to_numpy()
        while posicion - 1 >= 0 and valores[posicion - 1] > valores[posicion]:
            posicion -= 1
        if posicion < len(serie):
            descenso_desde = posicion
            truncado.iloc[posicion:] = True

    invalido = pd.Series(False, index=serie.index)
    for desde, hasta in huecos:
        invalido.iloc[desde : hasta + 1] = True
    invalido |= truncado

    validas = _rachas(~invalido)
    if not validas:
        raise ValueError("No queda ningún tramo con cobertura utilizable")
    desde, hasta = max(validas, key=lambda r: r[1] - r[0])
    recorte = serie.iloc[desde : hasta + 1]
    if len(recorte) < referencia:
        raise ValueError(
            f"La ventana de cobertura solo tiene {len(recorte)} días; "
            "revisa los parámetros antes de modelar"
        )

    audit = {
        "rule": {
            "hueco_minimo_dias": hueco_minimo,
            "fraccion_nivel": fraccion,
            "referencia_dias": referencia,
            "periodo": PERIODO,
        },
        "input": dict(serie.attrs.get("audit", {})),
        "coverage_gaps": [
            {
                "inicio": str(serie.index[a].date()),
                "fin": str(serie.index[b].date()),
                "dias": b - a + 1,
            }
            for a, b in huecos
        ],
        "first_truncated_day": (
            str(truncado[truncado].index[0].date()) if truncado.any() else None
        ),
        "final_descent_start": (
            str(serie.index[descenso_desde].date()) if descenso_desde is not None else None
        ),
        "window": {
            "inicio": str(recorte.index[0].date()),
            "fin": str(recorte.index[-1].date()),
            "dias": int(len(recorte)),
            "pedidos": int(recorte.sum()),
            "ceros": int((recorte == 0).sum()),
        },
        "discarded": {
            "dias_antes": int(desde),
            "dias_despues": int(len(serie) - hasta - 1),
            "pedidos_antes": int(serie.iloc[:desde].sum()),
            "pedidos_despues": int(serie.iloc[hasta + 1 :].sum()),
        },
        "legacy_window": {
            "inicio": VENTANA_HEREDADA[0],
            "fin": VENTANA_HEREDADA[1],
            "dias": int(len(serie.loc[VENTANA_HEREDADA[0] : VENTANA_HEREDADA[1]])),
            "pedidos": int(serie.loc[VENTANA_HEREDADA[0] : VENTANA_HEREDADA[1]].sum()),
        },
    }
    recorte = recorte.copy()
    recorte.attrs["audit"] = audit
    return Ventana(serie=recorte, inicio=recorte.index[0], fin=recorte.index[-1], audit=audit)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def _indice_futuro(train: pd.Series, steps: int) -> pd.DatetimeIndex:
    if steps < 1:
        raise ValueError("steps debe ser al menos 1")
    inicio = train.index[-1] + pd.Timedelta(days=1)
    return pd.date_range(inicio, periods=steps, freq="D")


def seasonal_naive(train: pd.Series, steps: int, periodo: int = PERIODO) -> pd.Series:
    """Repetir la última semana observada. Es el modo provisional del backend."""
    if len(train) < periodo:
        raise ValueError(f"Hacen falta al menos {periodo} días de historia")
    ultima = train.iloc[-periodo:].to_numpy()
    valores = [ultima[i % periodo] for i in range(steps)]
    return pd.Series(valores, index=_indice_futuro(train, steps), name="pronostico")


def weekday_mean(train: pd.Series, steps: int, semanas: int = 4) -> pd.Series:
    """Media de las últimas `semanas` semanas para cada día de la semana."""
    if len(train) < semanas * PERIODO:
        raise ValueError(f"Hacen falta al menos {semanas * PERIODO} días de historia")
    reciente = train.iloc[-semanas * PERIODO :]
    medias = reciente.groupby(reciente.index.day_of_week).mean()
    indice = _indice_futuro(train, steps)
    return pd.Series(medias.reindex(indice.day_of_week).to_numpy(), index=indice, name="pronostico")


# --------------------------------------------------------------------------- #
# Backtesting de origen móvil
# --------------------------------------------------------------------------- #
def rolling_origin(
    serie: pd.Series,
    *,
    n_origenes: int = 8,
    horizonte: int = 28,
    paso: int = 14,
    holdout: int = 28,
) -> list[tuple[pd.Series, pd.Series]]:
    """Particiones (train, test) de desarrollo, con ventana expansiva.

    Los últimos `holdout` días quedan fuera de todos los orígenes: son el
    conjunto reservado que solo se mira una vez, con el modelo ya congelado.
    """
    if holdout < 1 or horizonte < 1 or paso < 1 or n_origenes < 1:
        raise ValueError("Parámetros de backtesting inválidos")
    disponible = len(serie) - holdout
    minimo = horizonte * 2 + PERIODO
    particiones: list[tuple[pd.Series, pd.Series]] = []
    for k in range(n_origenes):
        fin_test = disponible - k * paso
        inicio_test = fin_test - horizonte
        if inicio_test < minimo:
            break
        train = serie.iloc[:inicio_test]
        test = serie.iloc[inicio_test:fin_test]
        if train.index.max() >= test.index.min():
            raise AssertionError("El origen usaría datos posteriores al corte")
        particiones.append((train, test))
    if not particiones:
        raise ValueError("La serie no da para ningún origen de desarrollo")
    return list(reversed(particiones))


def final_holdout(serie: pd.Series, *, holdout: int = 28) -> tuple[pd.Series, pd.Series]:
    """Train y test del conjunto reservado final."""
    if holdout < 1 or len(serie) <= holdout:
        raise ValueError("Holdout inválido para esta serie")
    return serie.iloc[:-holdout], serie.iloc[-holdout:]


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def _alineado(y: pd.Series, pred: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if len(y) != len(pred):
        raise ValueError(f"Longitudes distintas: {len(y)} y {len(pred)}")
    if not y.index.equals(pred.index):
        raise ValueError("El pronóstico no está alineado con el período evaluado")
    return y.to_numpy(dtype=float), pred.to_numpy(dtype=float)


def metrics(y: pd.Series, pred: pd.Series) -> dict:
    """MAE, RMSE, sMAPE y sesgo, en pedidos por día."""
    real, estimado = _alineado(y, pred)
    error = estimado - real
    denominador = (np.abs(real) + np.abs(estimado)) / 2
    smape = np.where(denominador > 0, np.abs(error) / np.where(denominador > 0, denominador, 1), 0.0)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "smape_pct": float(np.mean(smape) * 100),
        "bias": float(np.mean(error)),
        "dias": int(len(real)),
    }


def mase(y: pd.Series, pred: pd.Series, train: pd.Series, periodo: int = PERIODO) -> float:
    """MASE de Hyndman y Koehler: escala = MAE del naive estacional en train.

    Adimensional y sin el problema de los ceros que rompe el MAPE del script
    original, que divide por `test.clip(lower=1)`.
    """
    real, estimado = _alineado(y, pred)
    historia = train.to_numpy(dtype=float)
    if len(historia) <= periodo:
        raise ValueError("Train demasiado corto para escalar el MASE")
    escala = float(np.mean(np.abs(historia[periodo:] - historia[:-periodo])))
    if not escala > 0:
        raise ValueError("La escala del MASE es cero: train constante")
    return float(np.mean(np.abs(estimado - real)) / escala)


def rel_mae(y: pd.Series, pred: pd.Series, train: pd.Series, periodo: int = PERIODO) -> float:
    """MAE relativo al naive estacional **sobre el mismo test**.

    Vale exactamente 1 para el propio naive estacional, así que es la métrica
    interpretable del criterio de aceptación: por debajo de 1 el modelo bate a
    repetir la última semana; por encima, no.
    """
    real, estimado = _alineado(y, pred)
    base = seasonal_naive(train, len(y), periodo)
    base_mae = float(np.mean(np.abs(base.to_numpy(dtype=float) - real)))
    if not base_mae > 0:
        raise ValueError("El baseline no comete error: relación indefinida")
    return float(np.mean(np.abs(estimado - real)) / base_mae)


def coverage(y: pd.Series, inferior: pd.Series, superior: pd.Series) -> float:
    """Proporción de días reales dentro de la banda. Sin esto la banda decora."""
    real, bajo = _alineado(y, inferior)
    _, alto = _alineado(y, superior)
    if np.any(alto < bajo):
        raise ValueError("Banda invertida: superior por debajo de inferior")
    return float(np.mean((real >= bajo) & (real <= alto)))


# --------------------------------------------------------------------------- #
# Procedencia (para los JSON de métricas, con el esquema que ya usa el repo)
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    """SHA-256 de un fichero, leído por bloques."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as fichero:
        for bloque in iter(lambda: fichero.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def csv_hashes() -> dict:
    """SHA-256 del único CSV que usa este componente."""
    pedidos = OLIST / "olist_orders_dataset.csv"
    return {pedidos.name: sha256(pedidos)} if pedidos.exists() else {}


def code_hashes(*nombres: str) -> dict:
    """SHA-256 del código implicado, para poder rastrear qué produjo un número."""
    salida = {}
    for nombre in nombres:
        ruta = ROOT / nombre
        if ruta.exists():
            salida[nombre] = sha256(ruta)
    return salida


def versiones() -> dict:
    """Versiones reales del entorno, no las declaradas en requirements."""
    from importlib import metadata

    salida = {"python": sys.version.split()[0]}
    for paquete in ("pandas", "numpy", "statsmodels", "scikit-learn", "matplotlib", "joblib"):
        try:
            salida[paquete] = metadata.version(paquete)
        except metadata.PackageNotFoundError:
            pass
    return salida


def ahora_utc() -> str:
    """Marca temporal en UTC con segundos."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
