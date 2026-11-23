"""Particiones por compra, con madurez de etiquetas en cada fecha de corte."""
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    excluded: pd.DataFrame
    cutoffs: dict = field(default_factory=dict)

    def summary(self) -> dict:
        result = {"cutoffs": self.cutoffs}
        for name in ("train", "validation", "test"):
            frame = getattr(self, name)
            result[name] = {
                "orders": len(frame),
                "late": int(frame["retrasado"].sum()),
                "late_rate": float(frame["retrasado"].mean()),
                "purchase_min": str(frame["order_purchase_timestamp"].min()),
                "purchase_max": str(frame["order_purchase_timestamp"].max()),
                "latest_available_label": str(frame["label_available_at"].max()),
            }
        result["excluded"] = self.excluded["exclusion_reason"].value_counts().to_dict()
        return result


def chronological_split(
    frame: pd.DataFrame,
    train_end: str = "2018-05-01",
    validation_start: str = "2018-06-01",
    validation_end: str = "2018-07-01",
    test_start: str = "2018-08-01",
    test_end: str = "2018-09-01",
    observation_end: str = "2018-10-18",
) -> TemporalSplit:
    """Intervalos [inicio, fin); etiquetas estrictamente anteriores al corte.

    Train está maduro al comenzar validación; validación al comenzar prueba;
    prueba al cierre observacional. Las filas no utilizables se conservan aparte.
    """
    training_end, validation_cut, validation_end_cut, test_cut, end, observed = map(
        pd.Timestamp, (train_end, validation_start, validation_end, test_start, test_end, observation_end)
    )
    if not training_end <= validation_cut < validation_end_cut <= test_cut < end <= observed:
        raise ValueError("Cortes inválidos: train_end <= validation_start < validation_end <= test_start < test_end <= observation_end")
    if frame["order_id"].isna().any() or frame["order_id"].duplicated().any():
        raise ValueError("La partición requiere un registro por pedido")
    work = frame.reset_index(drop=True).copy()
    for column in ("order_purchase_timestamp", "label_available_at"):
        work[column] = pd.to_datetime(work[column], errors="raise")
        if work[column].isna().any():
            raise ValueError(f"Fecha faltante: {column}")
    purchase, available = work["order_purchase_timestamp"], work["label_available_at"]
    if available.lt(purchase).any():
        raise ValueError("La etiqueta no puede estar disponible antes de la compra")
    windows = {
        "train": purchase.lt(training_end),
        "validation": purchase.ge(validation_cut) & purchase.lt(validation_end_cut),
        "test": purchase.ge(test_cut) & purchase.lt(end),
    }
    cutoffs = {"train": validation_cut, "validation": test_cut, "test": observed}
    parts, reason = {}, pd.Series("outside_purchase_windows", index=work.index)
    reason.loc[purchase.ge(training_end) & purchase.lt(validation_cut)] = "train_label_maturation_gap"
    reason.loc[purchase.ge(validation_end_cut) & purchase.lt(test_cut)] = "validation_label_maturation_gap"
    for name, window in windows.items():
        mature = available.lt(cutoffs[name])
        reason.loc[window & ~mature] = f"{name}_label_not_available"
        selected = window & mature
        reason.loc[selected] = ""
        parts[name] = work.loc[selected].sort_values(
            ["order_purchase_timestamp", "order_id"], kind="stable"
        ).copy()
        if parts[name].empty:
            raise ValueError(f"Partición {name} vacía: revisa los cortes")
    excluded = work.loc[reason.ne("")].copy()
    excluded["exclusion_reason"] = reason.loc[reason.ne("")]
    if sum(len(part) for part in parts.values()) + len(excluded) != len(work):
        raise AssertionError("La partición no conserva el total de pedidos")
    return TemporalSplit(**parts, excluded=excluded, cutoffs={
        "train_purchase_end": str(training_end),
        "validation_start": str(validation_cut), "validation_end": str(validation_end_cut),
        "test_start": str(test_cut), "test_end": str(end), "observation_end": str(observed),
    })
