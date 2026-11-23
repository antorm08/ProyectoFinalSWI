"""Offline comparison of Olist delivery-duration regressors (not an API model)."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .common import MODELS, OLIST, ROOT
from .train_delay import CAT, NUM, build_dataset

TARGET = "dias_entrega"
FEATURES = NUM + CAT
DEV = (
    ("2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01"),
    ("2018-03-01", "2018-04-01", "2018-05-01", "2018-06-01"),
    ("2018-05-01", "2018-06-01", "2018-07-01", "2018-08-01"),
)
FINAL = ("2018-07-01", "2018-08-01", "2018-09-01", "2018-10-18")
DERIVED = {
    "precio_unitario": "precio / n_items",
    "flete_unitario": "flete / n_items",
    "peso_unitario": "peso_g / n_items",
    "flete_relativo": "flete / (precio + 0.01)",
}
REGIONS = {
    "centro_oeste": ["DF", "GO", "MS", "MT"],
    "nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "norte": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
    "sudeste": ["ES", "MG", "RJ", "SP"],
    "sur": ["PR", "RS", "SC"],
}
DERIVED["misma_region"] = " | ".join(
    f"((estado_cliente in {states!r}) & (estado_vendedor in {states!r}))" for states in REGIONS.values()
)
BASELINES = ("dummy_median", "olist_timestamp", "olist_calendar")
MODEL_NAMES = ("dummy_median", "ridge", "random_forest", "hgb", "hgb_derived", "hgb_without_promise")
BANDS = ((0, 5), (5, 10), (10, 15), (15, 30), (30, np.inf))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def time_stamp():
    return datetime.now(timezone.utc).isoformat()


def build_model(name):
    derived = name == "hgb_derived"
    nums = [x for x in NUM if name != "hgb_without_promise" or x != "dias_estimados"]
    if derived:
        nums += list(DERIVED)
    numeric = "passthrough"
    if name in ("ridge", "random_forest", "dummy_median"):
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        numeric = make_pipeline(imputer, StandardScaler()) if name == "ridge" else imputer
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", numeric, nums),
    ], remainder="drop")
    if name == "dummy_median":
        estimator = DummyRegressor(strategy="median")
    elif name == "ridge":
        estimator = Ridge(alpha=10.)
    elif name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=65, max_depth=14, min_samples_leaf=16, max_features=.70, n_jobs=2, random_state=42,
        )
    elif name.startswith("hgb") and name in MODEL_NAMES:
        estimator = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=220, max_leaf_nodes=31, min_samples_leaf=40,
            learning_rate=.07, l2_regularization=2., early_stopping=False, random_state=42,
        )
    elif name == "xgboost":
        from xgboost import XGBRegressor
        estimator = XGBRegressor(
            objective="reg:absoluteerror", n_estimators=220, max_depth=5, learning_rate=.07,
            subsample=.9, colsample_bytree=.85, reg_lambda=2., tree_method="hist", n_jobs=2, random_state=42,
        )
    else:
        raise ValueError(f"Modelo desconocido: {name}")
    steps = [prep, estimator]
    if derived:
        expr = "\n".join(f"{name} = {value}" for name, value in DERIVED.items())
        steps.insert(0, FunctionTransformer(pd.DataFrame.eval, kw_args={"expr": expr, "engine": "python"}))
    return make_pipeline(*steps)


def profile(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n": 0}
    return {"n": len(values), "mean": float(values.mean()),
            **dict(zip(("min", "p01", "p05", "p25", "median", "p75", "p90", "p95", "p99", "max"),
                       np.percentile(values, [0, 1, 5, 25, 50, 75, 90, 95, 99, 100]).tolist())),
            "over_30": int((values > 30).sum()), "over_60": int((values > 60).sum()),
            "over_90": int((values > 90).sum()), "below_one_day": int((values < 1).sum())}


def maturity_window(frame, start, end, cutoff):
    purchase = frame.order_purchase_timestamp
    in_window = purchase.lt(end)
    if start:
        in_window &= purchase.ge(start)
    mature = frame.label_available_at.lt(cutoff)
    include, omit = frame.loc[in_window & mature], frame.loc[in_window & ~mature]
    if include.empty:
        raise ValueError(f"Ventana sin etiquetas maduras: {start}, {end}, {cutoff}")
    include = include.sort_values(["order_purchase_timestamp", "order_id"], kind="stable")
    return include, {
        "purchase_start": start, "purchase_end": end, "label_cutoff": cutoff,
        "last_available_label": str(include.label_available_at.max()),
        "included": profile(include[TARGET]), "unavailable_retrospective": profile(omit[TARGET]),
    }


def split_for_cut(frame, cut):
    train_end, eval_start, eval_end, observe_end = cut
    if not train_end < eval_start < eval_end < observe_end:
        raise ValueError("Cortes no cronológicos o sin maduración")
    train, train_audit = maturity_window(frame, None, train_end, eval_start)
    evaluation, eval_audit = maturity_window(frame, eval_start, eval_end, observe_end)
    gap_n = int((frame.order_purchase_timestamp.ge(train_end) & frame.order_purchase_timestamp.lt(eval_start)).sum())
    return train, evaluation, {"train": train_audit, "evaluation": eval_audit, "gap_n": gap_n}


def promised_baselines(frame):
    return {
        "olist_timestamp": (frame.order_estimated_delivery_date - frame.order_purchase_timestamp)
                           .dt.total_seconds().to_numpy() / 86400,
        "olist_calendar": frame.dias_estimados.to_numpy(),
    }


def regression_metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("Predicciones/etiquetas no finitas o dimensiones distintas")
    if not len(y):
        return {"n": 0}
    residual = p - y
    ae = np.abs(residual)
    return {"n": len(y), "mae": float(ae.mean()), "rmse": float(np.sqrt(np.mean(residual**2))),
            "r2": float(r2_score(y, p)) if len(y) > 1 and np.ptp(y) > 0 else None,
            "medae": float(np.median(ae)), "within_2_pct": float(100 * (ae <= 2).mean()),
            "within_5_pct": float(100 * (ae <= 5).mean()), "bias_pred_minus_real": float(residual.mean())}


def breakdown(y, predictions):
    by_band = {}
    for lo, hi in BANDS:
        # Boundaries do not overlap; exactly 30 days still belongs to the 15–30 band.
        indices = (y <= hi) & ((y >= 0) if lo == 0 else (y > lo))
        by_band[f"{lo}-{hi}"] = {name: regression_metrics(y[indices], p[indices]) for name, p in predictions.items()}
    return by_band


def model_predict(model, frame):
    return np.maximum(model.predict(frame[FEATURES]), 0.)


def evaluate(train, evaluation, names):
    y = evaluation[TARGET].to_numpy()
    predictions = promised_baselines(evaluation)
    records = {name: regression_metrics(y, p) for name, p in predictions.items()}
    models = {}
    for name in names:
        pipeline = build_model(name)
        start = perf_counter()
        pipeline.fit(train[FEATURES], train[TARGET])
        fit_s = perf_counter() - start
        raw = pipeline.predict(evaluation[FEATURES])
        p = predictions[name] = np.maximum(raw, 0.)
        sample = evaluation[FEATURES].iloc[:1024]
        timings = []
        for _ in range(5):
            start = perf_counter()
            pipeline.predict(sample)
            timings.append((perf_counter() - start) * 1000)
        records[name] = {**regression_metrics(y, p), "fit_s": fit_s,
                         "predict_median_ms_per_batch": float(np.median(timings)), "latency_batch_n": len(sample),
                         "negative_raw_predictions": int((raw < 0).sum())}
        models[name] = pipeline
    return predictions, records, models


def macro(folds):
    keys = ("mae", "rmse", "r2", "medae", "within_2_pct", "within_5_pct", "bias_pred_minus_real")
    return {name: {key: float(np.mean([fold[name][key] for fold in folds])) for key in keys}
            for name in folds[0]}


def audit_dataset(frame):
    purchase, delivery = frame.order_purchase_timestamp, frame.order_delivered_customer_date
    approved = pd.to_datetime(frame.order_approved_at)
    carrier = pd.to_datetime(frame.order_delivered_carrier_date)
    conditions = {"approval_missing": approved.isna(), "carrier_missing": carrier.isna(),
                  "approval_before_purchase": approved.lt(purchase), "carrier_before_purchase": carrier.lt(purchase),
                  "carrier_before_approval": carrier.lt(approved), "delivery_before_approval": delivery.lt(approved),
                  "delivery_before_carrier": delivery.lt(carrier), "delivery_before_purchase": delivery.lt(purchase)}
    flag = np.logical_or.reduce([mask for key, mask in conditions.items() if not key.endswith("_missing")])
    return {"preparation": frame.attrs["audit"], "duration": profile(frame[TARGET]),
            "chronology": {key: int(mask.sum()) for key, mask in conditions.items()},
            "any_ordering_anomaly": profile(frame.loc[flag, TARGET]),
            "by_month": {str(period): profile(rows[TARGET])
                         for period, rows in frame.groupby(purchase.dt.to_period("M"))},
            "by_customer_state": {state: profile(rows[TARGET]) for state, rows in frame.groupby("estado_cliente")},
            "by_representative_seller_state": {state: profile(rows[TARGET]) for state, rows in frame.groupby("estado_vendedor")},
            "same_state": profile(frame.loc[frame.mismo_estado.eq(1), TARGET]),
            "different_state": profile(frame.loc[frame.mismo_estado.eq(0), TARGET])}


def bootstrap_deltas(frame, y, forecasts, selected, comparisons):
    days = frame.order_purchase_timestamp.dt.normalize().to_numpy()
    blocks = [np.flatnonzero(days == day) for day in np.unique(days)]
    rng = np.random.default_rng(42)
    errors = {name: np.abs(pred - y) for name, pred in forecasts.items() if name in {selected, *comparisons}}
    changes = {name: errors[selected] - errors[name] for name in comparisons}
    values = {name: [] for name in comparisons}
    for _ in range(500):
        sample = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        for name, delta in changes.items():
            values[name].append(delta[sample].mean())
    return {"days": len(blocks), "replicates": 500, "winner_minus_comparator_mae_ci95": {
        name: np.percentile(samples, [2.5, 97.5]).tolist() for name, samples in values.items()
    }}


def late_diagnostic(frame, prediction):
    # Midnight after the promised calendar date, expressed in fractional elapsed days.
    threshold = ((frame.order_estimated_delivery_date.dt.normalize() + pd.Timedelta(days=1)
                  - frame.order_purchase_timestamp).dt.total_seconds().to_numpy() / 86400)
    flagged = prediction >= threshold
    actual = frame.retrasado.to_numpy() == 1
    tp, fp, fn, tn = map(int, ((flagged & actual).sum(), (flagged & ~actual).sum(),
                               (~flagged & actual).sum(), (~flagged & ~actual).sum()))
    return {"confusion_tn_fp_fn_tp": [[tn, fp], [fn, tp]], "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None}


def render_plots(directory, result, y, predictions):
    from matplotlib import pyplot as plt

    winner = result["selection"]["model"]
    chosen, promised = predictions[winner], predictions["olist_timestamp"]
    names = sorted(result["development_macro"], key=lambda name: result["development_macro"][name]["mae"])

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / f"{name}.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.7, 5.2))
    density = ax.hexbin(y, chosen, gridsize=44, mincnt=1, bins="log", cmap="viridis")
    lim = max(y.max(), chosen.max())
    ax.plot([0, lim], [0, lim], "--", color="#a93425", label="predicción perfecta")
    ax.set(xlabel="Duración real (días)", ylabel="Predicción (días)", title=f"Agosto retrospectivo: real vs. {winner}")
    ax.legend()
    fig.colorbar(density, ax=ax, label="Pedidos (escala logarítmica)")
    save(fig, "real_vs_prediccion")

    fig, ax = plt.subplots(figsize=(7.7, 5.2))
    for label, p in ((winner, chosen), ("Plazo Olist exacto", promised)):
        ax.hist(p - y, bins=np.linspace(min((p - y).min() for p in (chosen, promised)),
                                      max((p - y).max() for p in (chosen, promised)), 55),
                histtype="step", linewidth=1.75, label=label)
    ax.axvline(0, ls="--", color="gray")
    ax.set(xlabel="Predicción menos entrega real (días)", ylabel="Pedidos", title="Distribución de errores, agosto")
    ax.legend()
    save(fig, "distribucion_errores")

    groups = result["august"]["by_duration"]
    tags = [f"{key.replace('-inf', '+')} (n={row[winner]['n']})" for key, row in groups.items()]
    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, name in enumerate((winner, "olist_timestamp", "dummy_median")):
        ax.bar(x + .24 * (i - 1), [row[name].get("mae", np.nan) for row in groups.values()], width=.23, label=name)
    ax.set(xticks=x, xticklabels=tags, ylabel="MAE (días)", title="Error por duración real, agosto")
    ax.legend()
    save(fig, "errores_por_duracion")

    fig, ax = plt.subplots(figsize=(9, 6))
    j = np.arange(len(names))
    ax.barh(j - .18, [result["development_macro"][n]["mae"] for n in names], .34, label="Desarrollo macro")
    ax.barh(j + .18, [result["august"]["metrics"][n]["mae"] for n in names], .34, label="Agosto retrospectivo")
    ax.set(yticks=j, yticklabels=names, xlabel="MAE (días)", title="Comparación de modelos y referencias")
    ax.invert_yaxis()
    ax.legend()
    save(fig, "comparacion_modelos")

    significance = result["permutation_importance"]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.barh([row["feature"] for row in significance], [row["mae_increase"] for row in significance],
            xerr=[row["std"] for row in significance], color="#287675")
    ax.axvline(0, color="gray", lw=.7)
    ax.set(xlabel="Incremento del MAE (días)",
           title=f"Importancia por permutación, {result['importance_model']} (desarrollo)")
    ax.invert_yaxis()
    save(fig, "importancia_permutacion")


def run(directory, artifact, require_xgboost=False):
    directory, artifact = Path(directory).resolve(), Path(artifact).resolve()
    if directory.exists() or artifact.exists() or artifact.name == "delay_model.joblib" or directory == artifact:
        raise FileExistsError("Use rutas nuevas y distintas de las rutas activas")
    names = list(MODEL_NAMES)
    try:
        import xgboost
    except (ImportError, OSError) as error:
        if require_xgboost:
            raise
        xgb_status = {"available": False, "reason": str(error)}
    else:
        names.append("xgboost")
        xgb_status = {"available": True, "version": xgboost.__version__}
    import matplotlib
    matplotlib.use("Agg")

    csv_before = {p.name: digest(p) for p in sorted(OLIST.glob("*.csv"))}
    artifacts_before = {p.name: digest(p) for p in sorted(MODELS.iterdir()) if p.is_file()}
    frame = build_dataset()
    np.testing.assert_allclose(frame[TARGET], (frame.order_delivered_customer_date - frame.order_purchase_timestamp)
                               .dt.total_seconds() / 86400, rtol=0, atol=0)
    development = frame.loc[frame.order_purchase_timestamp.lt(FINAL[1])]
    result = {"started_at_utc": time_stamp(), "target": TARGET, "features": FEATURES, "derivations": DERIVED,
              "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                           for name in ("pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "joblib")}},
              "xgboost": xgb_status, "csv_sha256": csv_before, "preexisting_model_sha256": artifacts_before,
              "code_sha256": {p: digest(ROOT / p) for p in ("ml/compare_delivery_regression.py", "ml/common.py", "ml/train_delay.py")},
              "development": []}
    with threadpool_limits(limits=2):
        for cut in DEV:
            train, evaluation, audit = split_for_cut(development, cut)
            forecasts, scores, _ = evaluate(train, evaluation, names)
            result["development"].append({"split": audit, "metrics": scores,
                                          "by_duration": breakdown(evaluation[TARGET].to_numpy(), forecasts)})
            print(f"Desarrollo completado: {cut[1]}", flush=True)
        result["development_macro"] = averages = macro([fold["metrics"] for fold in result["development"]])
        rank = sorted(averages, key=lambda name: (averages[name]["mae"], averages[name]["rmse"], name))
        winner = rank[0]
        result["selection"] = {"at_utc": time_stamp(), "model": winner, "ranking": rank,
                               "rule": "min macro development MAE among all contenders including baselines; then RMSE"}
        print(f"Selección congelada: {winner}", flush=True)
        if winner in BASELINES:
            importance_name = next(name for name in rank if name not in BASELINES)
        else:
            importance_name = winner
        train, valid, _ = split_for_cut(development, DEV[-1])
        model = build_model(importance_name).fit(train[FEATURES], train[TARGET])

        def neg_mae(estimator, inputs, target):
            return -np.abs(model_predict(estimator, inputs) - target.to_numpy()).mean()

        importance = permutation_importance(model, valid[FEATURES], valid[TARGET], scoring=neg_mae,
                                            n_repeats=5, random_state=42, n_jobs=1)
        result["permutation_importance"] = sorted(
            [{"feature": name, "mae_increase": float(mean), "std": float(std)}
             for name, mean, std in zip(FEATURES, importance.importances_mean, importance.importances_std)],
            key=lambda row: -row["mae_increase"])
        result["importance_model"] = importance_name
        train, test, audit = split_for_cut(frame, FINAL)
        predictions, scores, models = evaluate(train, test, names)
        y = test[TARGET].to_numpy()
        comparators = list(dict.fromkeys(n for n in rank[:4] + ["olist_timestamp", "dummy_median"] if n != winner))
        result["august"] = {"scored_at_utc": time_stamp(), "split": audit, "metrics": scores,
                            "by_duration": breakdown(y, predictions),
                            "late_calendar_rule_diagnostic": late_diagnostic(test, predictions[winner]),
                            "uncertainty": bootstrap_deltas(test, y, predictions, winner, comparators)}
        result["dataset_profile"] = audit_dataset(frame)
        binary = None
        if winner not in BASELINES:
            buffer = io.BytesIO()
            joblib.dump(models[winner], buffer, compress=3)
            binary = buffer.getvalue()
            loaded = joblib.load(io.BytesIO(binary))
            np.testing.assert_allclose(model_predict(loaded, test), predictions[winner], atol=1e-10, rtol=0)
        assert csv_before == {p.name: digest(p) for p in sorted(OLIST.glob("*.csv"))}
        assert artifacts_before == {p.name: digest(p) for p in sorted(MODELS.iterdir()) if p.is_file()}
    result["artifact"] = ({"name": artifact.name, "size": len(binary), "sha256": hashlib.sha256(binary).hexdigest(),
                           "roundtrip_absolute_tolerance_days": 1e-10, "activated": False} if binary else None)
    payload = json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False)
    directory.mkdir(parents=True, exist_ok=False)
    render_plots(directory, result, y, predictions)
    if binary:
        with artifact.open("xb") as handle:
            handle.write(binary)
    with (directory / "metricas.json").open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "resultados_tiempo_entrega")
    parser.add_argument("--artifact", type=Path, default=MODELS / "delivery_regression_experimental.joblib")
    parser.add_argument("--xgboost-path", type=Path, help="Directorio opcional de instalación local por pip --target")
    parser.add_argument("--require-xgboost", action="store_true")
    args = parser.parse_args()
    if args.xgboost_path:
        sys.path.insert(0, str(args.xgboost_path.resolve()))
    run(args.output, args.artifact, args.require_xgboost)
