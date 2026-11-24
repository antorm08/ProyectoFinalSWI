"""Shared rules for keeping reliable Olist history in training and inference."""
import numpy as np
import pandas as pd


def _runs(mask: pd.Series) -> list[tuple[int, int]]:
    values = mask.to_numpy(dtype=bool)
    borders = np.r_[0, np.flatnonzero(values[:-1] != values[1:]) + 1, len(values)]
    return [(int(start), int(stop - 1)) for start, stop in zip(borders[:-1], borders[1:]) if values[start]]


def coverage_bounds(serie: pd.Series, *, hueco_minimo=2, fraccion=0.25, referencia=180):
    """Return inclusive positions, gaps and cutoff; do not impose a training minimum."""
    if serie.empty:
        raise ValueError("No hay histórico de demanda")
    gaps = [r for r in _runs(serie.eq(0)) if r[1] - r[0] + 1 >= hueco_minimo]
    weekly = serie.rolling(7).sum()
    normal = weekly.rolling(referencia).median().shift(7)
    truncated = weekly.lt(fraccion * normal).fillna(False)
    descent = None
    if truncated.any():
        descent = int(np.flatnonzero(truncated.to_numpy())[0])
        values = weekly.to_numpy()
        while descent > 0 and values[descent - 1] > values[descent]:
            descent -= 1
        truncated.iloc[descent:] = True

    invalid = truncated.copy()
    for start, end in gaps:
        invalid.iloc[start:end + 1] = True
    valid = _runs(~invalid)
    if not valid:
        raise ValueError("No queda ningún tramo con cobertura utilizable")
    start, end = max(valid, key=lambda r: r[1] - r[0])
    return start, end, gaps, descent
