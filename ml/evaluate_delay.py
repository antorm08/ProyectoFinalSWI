"""Experimento exploratorio de retrasos; prueba final solo después de congelar umbral.

No activa automáticamente el candidato en la API ni sobrescribe archivos.
Ejecutar desde la raíz: python -B -m ml.evaluate_delay
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from threadpoolctl import threadpool_limits

from .common import MODELS, OLIST, ROOT
from .temporal import chronological_split
from .train_delay import CAT, NUM, build_dataset, build_pipeline

FEATURES = NUM + CAT
THRESHOLDS = [float(value) for value in np.arange(1, 100) / 100]
SEED = 42


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def metrics(y, probabilities, threshold=0.5):
    y, probabilities = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    if len(y) != len(probabilities):
        raise ValueError("Tamaños distintos de etiquetas y scores")
    if not len(y):
        return {"n": 0, "reason": "subgrupo vacío"}
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Scores fuera de [0, 1] o no finitos")
    predicted = probabilities >= threshold
    precision, recall, _ = precision_recall_curve(y, probabilities) if y.sum() else (None, None, None)
    return {
        "n": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "threshold": float(threshold), "predicted_positive": int(predicted.sum()),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, probabilities)) if y.sum() else None,
        "pr_auc_trapezoid": float(auc(recall, precision)) if y.sum() else None,
        "brier": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "mean_score": float(probabilities.mean()),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def select_threshold(y_validation, p_validation):
    """F1 positivo máximo en grilla fija; empate por precisión y umbral mayor."""
    rows = []
    for threshold in THRESHOLDS:
        pred = np.asarray(p_validation) >= threshold
        rows.append({
            "threshold": threshold,
            "precision": float(precision_score(y_validation, pred, zero_division=0)),
            "recall": float(recall_score(y_validation, pred, zero_division=0)),
            "f1": float(f1_score(y_validation, pred, zero_division=0)),
            "alerts": int(pred.sum()),
        })
    selected = max(rows, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    return selected["threshold"], rows


def reliability(y, p):
    """Diez bandas fijas. Los intervalos vacíos no aportan un ECE inventado."""
    y, p = np.asarray(y), np.asarray(p)
    bins, ece = [], 0.
    assigned = np.minimum((p * 10).astype(int), 9)
    for index in range(10):
        mask = assigned == index
        count = int(mask.sum())
        score, rate = (float(p[mask].mean()), float(y[mask].mean())) if count else (None, None)
        bins.append({"lower": index / 10, "upper": (index + 1) / 10, "n": count,
                     "mean_score": score, "observed_rate": rate})
        if count:
            ece += count / len(y) * abs(score - rate)
    return {"ece_10_uniform_bins": float(ece), "bins": bins}


def undo_class_weights(p, prevalence_train):
    """Diagnóstico algebraico bajo supuestos ideales; NO calibración validada."""
    ratio = (1 - prevalence_train) / prevalence_train
    p = np.asarray(p)
    return p / (p + ratio * (1 - p))


def anomaly_mask(frame):
    actual = frame["order_delivered_customer_date"]
    purchase = frame["order_purchase_timestamp"]
    approved = pd.to_datetime(frame["order_approved_at"], errors="raise")
    carrier = pd.to_datetime(frame["order_delivered_carrier_date"], errors="raise")
    return carrier.lt(purchase) | carrier.lt(approved) | actual.lt(approved) | actual.lt(carrier)


def cohorts(frame):
    return {
        "all": np.ones(len(frame), dtype=bool),
        "delivery_le_30d": frame["dias_entrega"].le(30).to_numpy(),
        "delivery_gt_30d": frame["dias_entrega"].gt(30).to_numpy(),
        "delivery_gt_60d": frame["dias_entrega"].gt(60).to_numpy(),
        "chronology_anomaly": anomaly_mask(frame).to_numpy(),
        "no_chronology_anomaly": (~anomaly_mask(frame)).to_numpy(),
        "multiple_origin_states": frame["n_estados_vendedor"].gt(1).to_numpy(),
        "unknown_weight": frame["peso_g"].isna().to_numpy(),
        "same_representative_state": frame["mismo_estado"].eq(1).to_numpy(),
        "different_representative_state": frame["mismo_estado"].eq(0).to_numpy(),
    }


def cohort_metrics(frame, p, threshold):
    y = frame["retrasado"].to_numpy()
    return {name: metrics(y[mask], p[mask], threshold) for name, mask in cohorts(frame).items()}


async def asgi_post(application, body):
    """Ejercita la ruta HTTP real en proceso, sin dependencias adicionales."""
    sent = []
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/api/delay/predict", "raw_path": b"/api/delay/predict",
        "query_string": b"", "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 10000), "server": ("testserver", 80), "root_path": "",
    }
    await application(scope, receive, send)
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    content = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    return status, json.loads(content)


def check_compatibility(model, path, sample):
    sys.path.insert(0, str(ROOT / "backend"))
    from app.main import app
    from app.schemas import DelayInput
    from app.services import delay_service

    sample = sample.copy()
    expected = model.predict_proba(sample[FEATURES])[:, 1]
    results = []
    with patch.object(delay_service, "DELAY_MODEL", Path(path)), patch.object(delay_service, "_model", None):
        for (_, row), score in zip(sample.iterrows(), expected):
            payload = {
                "precio": float(row.precio), "flete": float(row.flete),
                "peso_g": 0. if pd.isna(row.peso_g) else float(row.peso_g),
                "n_items": int(row.n_items), "dias_estimados": int(row.dias_estimados),
                "estado_cliente": str(row.estado_cliente).lower(),
                "estado_vendedor": str(row.estado_vendedor).lower(),
                "mes": int(row.mes), "dia_semana": int(row.dia_semana),
            }
            direct = delay_service.predict(DelayInput(**payload)).model_dump()
            status, http = asyncio.run(asgi_post(app, payload))
            assert status == 200 and direct == http
            assert set(http) == {"probabilidad_retraso", "nivel_riesgo", "modelo_entrenado"}
            assert http["modelo_entrenado"] and http["probabilidad_retraso"] == round(float(score), 4)
            expected_level = "alto" if score >= .5 else "medio" if score >= .2 else "bajo"
            assert http["nivel_riesgo"] == expected_level
            results.append({"status": status, "matches_pipeline": True,
                            "unknown_weight": bool(pd.isna(row.peso_g)), "multiple_states": int(row.n_estados_vendedor)})
        bad_status, _ = asyncio.run(asgi_post(app, {**payload, "precio": -1}))
        assert bad_status == 422
    return {"real_service_and_asgi_route": results, "invalid_input_status": bad_status,
            "patched_path_only_during_test": True, "active_model_unchanged": True}


def run(model_path=None, metrics_path=None):
    model_path = Path(model_path or MODELS / "delay_model_exploratorio.joblib")
    metrics_path = Path(metrics_path or ROOT / "docs" / "evaluacion_retrasos_metricas.json")
    for destination in (model_path, metrics_path):
        if destination.exists():
            raise FileExistsError(f"No se sobrescribe: {destination}")
    start = time.perf_counter()
    raw_files = {path.name: sha256(path) for path in sorted(OLIST.glob("*.csv"))}
    prior_artifacts = {path.name: sha256(path) for path in MODELS.iterdir() if path.is_file()}
    frame = build_dataset()
    split = chronological_split(frame)
    assert len(frame) == 96470 and int(frame["retrasado"].sum()) == 6534
    assert set(FEATURES) == {
        "precio", "flete", "peso_g", "n_items", "dias_estimados", "mes", "dia_semana",
        "mismo_estado", "estado_cliente", "estado_vendedor",
    }
    assert split.train["label_available_at"].lt("2018-06-01").all()
    assert split.validation["label_available_at"].lt("2018-08-01").all()
    assert split.test["label_available_at"].lt("2018-10-18").all()
    for part in (split.train, split.validation, split.test):
        assert part["retrasado"].nunique() == 2
    result = {
        "started_at_utc": now(),
        "protocol": {
            "features": FEATURES, "seed": SEED, "threads": 1,
            "parameters": build_pipeline().named_steps["clf"].get_params(),
            "threshold_grid": THRESHOLDS,
            "threshold_selection": "max validation F1; ties by precision then higher threshold",
            "baseline": "DummyClassifier(strategy=prior), trained only on train",
            "main_candidate": "original weighted HGB; no probability calibration applied",
            "sensitivity_changes_candidate": False,
            "final_test": "August; only scored after all selection frozen",
            "long_delivery_thresholds_days": [30, 60],
            "future_label_cohorts": "post-hoc diagnostics only, after freezing and final test",
        },
        "versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl")
        }},
        "source_csv_sha256": raw_files,
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            ROOT / "ml/common.py", ROOT / "ml/temporal.py",
            ROOT / "ml/train_delay.py", Path(__file__).resolve(),
            ROOT / "backend/app/services/delay_service.py",
        )},
        "preparation": frame.attrs["audit"], "split": split.summary(),
        "fit_order_ids_sha256": hashlib.sha256("\n".join(sorted(split.train.order_id)).encode()).hexdigest(),
        "label_exclusions": split.excluded.groupby("exclusion_reason").agg(
            orders=("order_id", "size"), late=("retrasado", "sum"),
            duration_mean=("dias_entrega", "mean"), duration_max=("dias_entrega", "max"),
        ).to_dict(orient="index"),
    }
    y_train, y_val = split.train["retrasado"].to_numpy(), split.validation["retrasado"].to_numpy()
    with threadpool_limits(limits=1):
        model = build_pipeline()
        model.fit(split.train[FEATURES], y_train)
        p_train = model.predict_proba(split.train[FEATURES])[:, 1]
        p_val = model.predict_proba(split.validation[FEATURES])[:, 1]
        threshold, curve = select_threshold(y_val, p_val)
        result["validation_threshold_curve"] = curve
        baseline = DummyClassifier(strategy="prior", random_state=SEED)
        baseline.fit(np.zeros((len(y_train), 1)), y_train)
        baseline_val = baseline.predict_proba(np.zeros((len(y_val), 1)))[:, 1]
        baseline_threshold, baseline_curve = select_threshold(y_val, baseline_val)
        result["baseline_validation_threshold_curve"] = baseline_curve
        result["development_metrics"] = {
            name: {"at_0_5": metrics(y, p, .5), "at_validation_threshold": metrics(y, p, threshold)}
            for name, y, p in [("train_apparent", y_train, p_train), ("validation", y_val, p_val)]
        }
        corrected_val = undo_class_weights(p_val, float(y_train.mean()))
        result["calibration_before_final_test"] = {
            "raw_validation": reliability(y_val, p_val),
            "undo_training_weights_diagnostic": metrics(y_val, corrected_val),
            "diagnostic_not_selected_or_saved": True,
            "reason_no_calibrator_fit": "Only 51 validation positives, censored outcomes and major prevalence drift; no reliable production calibration claim.",
        }
        result["validation_error_cohorts"] = cohort_metrics(split.validation, p_val, threshold)
        # Sensibilidades de calidad: no modifican al candidato principal.
        sensitivity = {}
        for name, keep in {
            "exclude_train_chronology_anomalies": ~anomaly_mask(split.train),
            "exclude_train_delivery_gt_60d": split.train["dias_entrega"].le(60),
        }.items():
            subset = split.train.loc[keep]
            variant = build_pipeline()
            variant.fit(subset[FEATURES], subset["retrasado"])
            p_variant = variant.predict_proba(split.validation[FEATURES])[:, 1]
            sensitivity[name] = {
                "training_orders": len(subset), "training_late": int(subset["retrasado"].sum()),
                "validation_at_main_validation_threshold": metrics(y_val, p_variant, threshold),
                "validation_at_0_5": metrics(y_val, p_variant, .5),
                "threshold_not_retuned": True, "not_candidate": True,
            }
        result["training_sensitivity"] = sensitivity
        # Tres orígenes anteriores, sin leer resultados de agosto para decidir.
        folds = [
            ("2017-11-01", "2017-12-01", "2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01"),
            ("2017-12-01", "2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01", "2018-05-01"),
            ("2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01", "2018-05-01", "2018-06-01"),
        ]
        earlier = frame.loc[frame["order_purchase_timestamp"].lt("2018-08-01")]
        fold_results = []
        for train_end, val_start, val_end, test_start, test_end, observed in folds:
            fold = chronological_split(earlier, train_end, val_start, val_end, test_start, test_end, observed)
            variant = build_pipeline()
            variant.fit(fold.train[FEATURES], fold.train["retrasado"])
            p_fold = variant.predict_proba(fold.validation[FEATURES])[:, 1]
            fold_results.append({
                "fit_purchase_end": train_end, "fit_label_cutoff": val_start,
                "evaluation_purchase_start": val_start, "evaluation_purchase_end": val_end,
                "evaluation_label_cutoff": test_start,
                "train_n": len(fold.train), "train_late": int(fold.train["retrasado"].sum()),
                "evaluation_at_predefined_0_5": metrics(fold.validation["retrasado"], p_fold),
                "eval_unobserved_at_cutoff": int(fold.excluded["exclusion_reason"].eq("validation_label_not_available").sum()),
                "same_parameters": True, "not_used_to_choose_model_or_threshold": True,
            })
        result["earlier_temporal_folds"] = fold_results
        importance = permutation_importance(
            model, split.validation[FEATURES], y_val, scoring="average_precision",
            n_repeats=3, random_state=SEED, n_jobs=1,
        )
        result["validation_permutation_importance_ap"] = sorted([
            {"feature": feature, "mean_ap_drop": float(mean), "std": float(std)}
            for feature, mean, std in zip(FEATURES, importance.importances_mean, importance.importances_std)
        ], key=lambda row: -row["mean_ap_drop"])
        duplicate = clone(model)
        duplicate.fit(split.train[FEATURES], y_train)
        repeat_val = duplicate.predict_proba(split.validation[FEATURES])[:, 1]
        repeat_train = duplicate.predict_proba(split.train[FEATURES].iloc[:1024])[:, 1]
        max_difference = float(max(np.max(np.abs(p_val - repeat_val)), np.max(np.abs(p_train[:1024] - repeat_train))))
        assert max_difference <= 1e-12, f"No reproducible: {max_difference}"
        result["reproducibility"] = {
            "independent_second_fit_same_train": True, "max_abs_score_difference": max_difference,
            "tolerance": 1e-12, "not_fit_on_validation_or_test": True,
        }
        sample = pd.concat([
            split.validation.head(3),
            split.train.loc[split.train["peso_g"].isna()].head(2),
            split.train.loc[split.train["n_estados_vendedor"].gt(1)].head(2),
            split.train.loc[split.train["n_items"].gt(1)].head(2),
        ]).drop_duplicates("order_id")
        with tempfile.TemporaryDirectory(prefix="commerceai-delay-", dir=r"C:\Users\MSI\AppData\Local\Temp\opencode") as temporary:
            temporary_artifact = Path(temporary) / "candidate.joblib"
            joblib.dump(model, temporary_artifact, compress=3)
            loaded = joblib.load(temporary_artifact)
            np.testing.assert_allclose(loaded.predict_proba(split.validation[FEATURES])[:, 1], p_val, rtol=0, atol=0)
            result["compatibility"] = check_compatibility(model, temporary_artifact, sample)
            # Esta decisión se congela antes de ejecutar predict_proba sobre prueba.
            result["frozen_selection"] = {
                "frozen_at_utc": now(), "model": "weighted HGB, unchanged",
                "threshold": threshold, "baseline_threshold": baseline_threshold,
                "score": "uncalibrated raw predict_proba", "fit_n": len(split.train),
            }
            print(json.dumps({"frozen_selection": result["frozen_selection"]}), flush=True)
            y_test = split.test["retrasado"].to_numpy()
            p_test = model.predict_proba(split.test[FEATURES])[:, 1]
            result["final_test_scored_at_utc"] = now()
            result["final_test"] = {
                "at_0_5": metrics(y_test, p_test),
                "at_validation_threshold": metrics(y_test, p_test, threshold),
                "service_0_2_diagnostic_predefined": metrics(y_test, p_test, .2),
                "thresholds_not_retuned": True,
            }
            result["baseline"] = {}
            for name, y in [("train_apparent", y_train), ("validation", y_val), ("final_test", y_test)]:
                p_baseline = baseline.predict_proba(np.zeros((len(y), 1)))[:, 1]
                result["baseline"][name] = {
                    "at_0_5": metrics(y, p_baseline),
                    "at_baseline_validation_threshold": metrics(y, p_baseline, baseline_threshold),
                }
            result["final_test_calibration_diagnostic_only"] = reliability(y_test, p_test)
            result["final_test_error_cohorts_diagnostic_only"] = cohort_metrics(split.test, p_test, threshold)
            # Etiquetas tardías solo se inspeccionan DESPUÉS de congelar y evaluar.
            future_diagnostics = {}
            for reason in ("train_label_not_available", "validation_label_not_available"):
                cohort = split.excluded.loc[split.excluded["exclusion_reason"].eq(reason)]
                if len(cohort):
                    future_diagnostics[reason] = metrics(
                        cohort["retrasado"], model.predict_proba(cohort[FEATURES])[:, 1], threshold
                    )
            full_june = frame.loc[
                frame["order_purchase_timestamp"].ge("2018-06-01")
                & frame["order_purchase_timestamp"].lt("2018-07-01")
            ]
            future_diagnostics["complete_june_not_for_selection"] = metrics(
                full_june["retrasado"], model.predict_proba(full_june[FEATURES])[:, 1], threshold
            )
            result["posthoc_long_delivery_labels_not_for_selection"] = future_diagnostics
            assert raw_files == {path.name: sha256(path) for path in sorted(OLIST.glob("*.csv"))}
            assert prior_artifacts == {path.name: sha256(path) for path in MODELS.iterdir() if path.is_file()}
            json.dumps(result, ensure_ascii=False, allow_nan=False)
            # Copia exclusiva: jamás sobreescribir modelos preexistentes.
            with model_path.open("xb") as handle:
                handle.write(temporary_artifact.read_bytes())
    result["artifact"] = {
        "path": str(model_path.relative_to(ROOT)), "sha256": sha256(model_path),
        "bytes": model_path.stat().st_size, "active_in_default_service": False,
        "uncalibrated": True, "production_ready": False,
    }
    result["csv_hashes_unchanged"] = True
    result["previous_artifacts_unchanged"] = True
    result["completed_at_utc"] = now()
    result["elapsed_seconds"] = time.perf_counter() - start
    with metrics_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({
        "metrics_file": str(metrics_path), "artifact": result["artifact"],
        "validation": result["development_metrics"]["validation"],
        "final_test": result["final_test"],
    }, ensure_ascii=True, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--metrics-path", type=Path)
    args = parser.parse_args()
    run(args.model_path, args.metrics_path)
