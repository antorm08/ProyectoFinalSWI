"""Calendario conocido de antemano para el pronóstico de demanda (componente 2).

Las exógenas de un pronóstico solo valen si se conocen antes de pronosticar. Las
fiestas de Brasil y los Black Friday cumplen esa condición, así que se declaran
como datos en lugar de calcularse: las móviles (Carnaval, Viernes Santo y Corpus
Christi) dependen de la Pascua y no hay forma legible de derivarlas sin añadir una
dependencia nueva al proyecto.

Este módulo vive en el backend y lo importan también los scripts de `ml/`, igual
que `ml/evaluate_products.py` importa `app.services.product_service`. Así el
artefacto entrenado y el servicio construyen exactamente las mismas columnas.
"""
from __future__ import annotations

import pandas as pd

#: Columnas que devuelve `calendar_exog`, en este orden y siempre todas.
COLUMNAS: tuple[str, ...] = ("festivo", "black_friday", "black_friday_cola")

#: Años para los que el calendario está declarado. Fuera de este rango no se
#: devuelven ceros: se falla, porque un cero silencioso sería una fiesta perdida.
COBERTURA: tuple[int, int] = (2016, 2020)

#: Fiestas nacionales de fecha fija (mes, día).
FIJAS: tuple[tuple[int, int], ...] = (
    (1, 1),    # Confraternización Universal
    (4, 21),   # Tiradentes
    (5, 1),    # Día del Trabajo
    (9, 7),    # Independencia
    (10, 12),  # Nuestra Señora Aparecida
    (11, 2),   # Finados
    (11, 15),  # Proclamación de la República
    (12, 25),  # Navidad
)

#: Fiestas nacionales móviles: Carnaval (lunes y martes), Viernes Santo y Corpus
#: Christi. Declaradas año a año.
MOVILES: tuple[str, ...] = (
    "2016-02-08", "2016-02-09", "2016-03-25", "2016-05-26",
    "2017-02-27", "2017-02-28", "2017-04-14", "2017-06-15",
    "2018-02-12", "2018-02-13", "2018-03-30", "2018-05-31",
    "2019-03-04", "2019-03-05", "2019-04-19", "2019-06-20",
    "2020-02-24", "2020-02-25", "2020-04-10", "2020-06-11",
)

#: Viernes de Black Friday. En Olist el 2017-11-24 marca 1.176 pedidos frente a
#: una mediana diaria de 151,5 (docs/validacion_olist.md, sección 5.2).
BLACK_FRIDAY: tuple[str, ...] = (
    "2016-11-25", "2017-11-24", "2018-11-23", "2019-11-29", "2020-11-27",
)

#: Días posteriores al viernes que conservan demanda elevada. En 2017 los cuatro
#: máximos siguientes de toda la serie son los días 25, 26, 27 y 28 de noviembre.
COLA_DIAS: int = 4


class CalendarioFueraDeRango(ValueError):
    """El índice pedido se sale de los años declarados en COBERTURA."""


def _fechas_fijas(años: range) -> set[pd.Timestamp]:
    return {pd.Timestamp(year=a, month=m, day=d) for a in años for m, d in FIJAS}


def _fechas_cola() -> set[pd.Timestamp]:
    cola: set[pd.Timestamp] = set()
    for viernes in BLACK_FRIDAY:
        inicio = pd.Timestamp(viernes)
        cola.update(inicio + pd.Timedelta(days=n) for n in range(1, COLA_DIAS + 1))
    return cola


def calendar_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Matriz de exógenas de calendario para `index`, con `COLUMNAS` en float.

    Determinista y sin estado: el mismo índice devuelve siempre lo mismo, tanto
    sobre el histórico al entrenar como sobre fechas futuras al pronosticar.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("calendar_exog necesita un DatetimeIndex")
    if index.empty:
        return pd.DataFrame(0.0, index=index, columns=list(COLUMNAS))
    primero, ultimo = index.min().year, index.max().year
    if primero < COBERTURA[0] or ultimo > COBERTURA[1]:
        raise CalendarioFueraDeRango(
            f"El calendario está declarado para {COBERTURA[0]}–{COBERTURA[1]}; "
            f"se pidió {primero}–{ultimo}. Amplía FIJAS, MOVILES y BLACK_FRIDAY."
        )
    dias = index.normalize()
    festivos = _fechas_fijas(range(primero, ultimo + 1)) | {pd.Timestamp(f) for f in MOVILES}
    viernes = {pd.Timestamp(f) for f in BLACK_FRIDAY}
    exog = pd.DataFrame(
        {
            "festivo": dias.isin(festivos),
            "black_friday": dias.isin(viernes),
            "black_friday_cola": dias.isin(_fechas_cola()),
        },
        index=index,
    ).astype(float)
    return exog[list(COLUMNAS)]
