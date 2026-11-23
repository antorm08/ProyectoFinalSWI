"""Temporal development and frozen retrospective comparison for delay prediction."""
import argparse
import io
import json
import platform
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from .common import MODELS, OLIST, ROOT
from .evaluate_delay import FEATURES, check_compatibility, metrics, now, reliability, sha256
from .train_delay import NUM, build_dataset, build_pipeline

OLD = MODELS / "delay_model_exploratorio.joblib"
OLD_THRESHOLD = .53
ACTIVE = MODELS / "delay_model.joblib"
DEV = (
    ("2017-09-01", "2017-10-01", "2017-12-01", "2018-01-01", "2018-02-01", "2018-03-01"),
    ("2017-11-01", "2017-12-01", "2018-02-01", "2018-03-01", "2018-04-01", "2018-05-01"),
    ("2018-01-01", "2018-02-01", "2018-04-01", "2018-05-01", "2018-06-01", "2018-07-01"),
)
FINAL = ("2018-04-01", "2018-05-01", "2018-07-01", "2018-08-01", "2018-09-01", "2018-10-18")
VARIANTS = {
    # (weights, compact HGB, derived variables). Logistic is a separate family.
    "logistic": (None, False, False),
    "logistic_balanced": ("balanced", False, False),
    "hgb_original_config": ("balanced", False, False),
    "hgb_unweighted": (None, False, False),
    "hgb_compact_unweighted": (None, True, False),
    "hgb_compact_weight3": ({0: 1, 1: 3}, True, False),
    "hgb_compact_balanced": ("balanced", True, False),
    "hgb_compact_unweighted_derived": (None, True, True),
    "hgb_compact_weight3_derived": ({0: 1, 1: 3}, True, True),
}
CALIBRATIONS = ("raw", "sigmoid", "isotonic")
DERIVED = {
    "freight_share": "flete / (precio + flete + 0.01)",
    "price_per_item": "precio / n_items",
    "weight_per_item": "peso_g / n_items",
    "free_freight": "flete == 0",
    "weekend": "dia_semana >= 5",
}
ALERT_BUDGET = .10


def build_variant(name):
    weights, compact, derived = VARIANTS[name]
    base = build_pipeline()
    if name.startswith("logistic"):
        base.set_params(
            prep__num=make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
            clf=LogisticRegression(class_weight=weights, C=1., max_iter=1000, random_state=42),
        )
    else:
        base.set_params(clf__class_weight=weights)
        if compact:
            base.set_params(clf__max_iter=150, clf__max_leaf_nodes=15, clf__min_samples_leaf=80,
                            clf__l2_regularization=2., clf__learning_rate=.07)
    if not derived:
        return base
    base.set_params(prep__transformers=[
        (key, transformer, NUM + list(DERIVED) if key == "num" else columns)
        for key, transformer, columns in base.named_steps["prep"].transformers
    ])
    # Standard sklearn/pandas objects only, so joblib requires no local ML class.
    expression = "\n".join(f"{key} = {value}" for key, value in DERIVED.items())
    return make_pipeline(FunctionTransformer(pd.DataFrame.eval, kw_args={"expr": expression, "engine": "python"}), base)


def window(frame, start, end, deadline):
    purchase, available = frame["order_purchase_timestamp"], frame["label_available_at"]
    in_window = purchase.lt(end)
    if start is not None:
        in_window &= purchase.ge(start)
    active = in_window & available.lt(deadline)
    omitted = in_window & ~active
    part = frame.loc[active].sort_values(["order_purchase_timestamp", "order_id"])
    if part["retrasado"].nunique() != 2:
        raise ValueError(f"Ventana sin ambas clases: {start}, {end}, {deadline}")
    return part, {
        "purchase_start": start, "purchase_end": end, "label_cutoff": deadline,
        "last_label": str(part.label_available_at.max()),
        "n": len(part), "late": int(part.retrasado.sum()), "prevalence": float(part.retrasado.mean()),
        "unavailable_n": int(omitted.sum()), "unavailable_late_retrospective": int(frame.loc[omitted, "retrasado"].sum()),
    }


def development_windows(frame, cuts):
    train_end, cal_start, cal_end, eval_start, eval_end, deadline = cuts
    if not train_end < cal_start < cal_end < eval_start < eval_end < deadline:
        raise ValueError("Ventanas solapadas, sin maduración o fuera de orden")
    train, train_audit = window(frame, None, train_end, cal_start)
    cal, cal_audit = window(frame, cal_start, cal_end, eval_start)
    evaluation, eval_audit = window(frame, eval_start, eval_end, deadline)
    return train, cal, evaluation, {"fit": train_audit, "calibration": cal_audit, "evaluation": eval_audit}


def fit_base(name, train):
    return build_variant(name).fit(train[FEATURES], train.retrasado)


def calibrate(base, method, cal):
    if method == "raw":
        return base
    # Official frozen estimator: calibration cannot modify the fitted base.
    return CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(cal[FEATURES], cal.retrasado)


def scores(model, frame):
    return model.predict_proba(frame[FEATURES])[:, 1]


def choose_threshold(y, p, budget=ALERT_BUDGET):
    """Max F1 subject to alert cap; handle score ties without splitting them."""
    order = np.argsort(-p, kind="stable")
    ordered = p[order]
    cut = np.flatnonzero(np.r_[ordered[:-1] != ordered[1:], True])
    count = cut + 1
    tp = np.cumsum(y[order])[cut]
    choices = [(float(2 * found / (alerts + y.sum())), float(found / alerts), float(ordered[end]))
               for end, alerts, found in zip(cut, count, tp) if alerts <= budget * len(y)]
    if not choices or max(choices)[0] == 0:
        return float(np.nextafter(1., np.inf))
    return max(choices)[2]


def aggregate(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in
            ("average_precision", "roc_auc", "brier", "log_loss")}


def paired_day_bootstrap(test, new, old, new_threshold, replicates=500):
    """Paired purchase-day bootstrap, conditional on this already observed month."""
    rng = np.random.default_rng(42)
    days = test.order_purchase_timestamp.dt.normalize().to_numpy()
    blocks = [np.flatnonzero(days == day) for day in np.unique(days)]
    y = test.retrasado.to_numpy()
    keys = ("average_precision", "roc_auc", "brier", "f1", "recall")
    deltas = []
    for _ in range(replicates):
        sampled = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        if len(np.unique(y[sampled])) < 2:
            continue
        a = metrics(y[sampled], new[sampled], new_threshold)
        b = metrics(y[sampled], old[sampled], OLD_THRESHOLD)
        deltas.append([a[key] - b[key] for key in keys])
    return {
        "purchase_days": len(blocks), "replicates_used": len(deltas),
        "percentile_95_new_minus_old": dict(zip(keys, np.percentile(deltas, [2.5, 97.5], axis=0).T.tolist())),
    }


def run(output, artifact):
    output, artifact = Path(output).resolve(), Path(artifact).resolve()
    for path in (output, artifact):
        if path == ACTIVE.resolve() or path.exists():
            raise FileExistsError(f"Destino reservado o existente: {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(path.parent)
    if output == artifact:
        raise ValueError("Los destinos deben ser distintos")
    previous = {p.name: sha256(p) for p in MODELS.iterdir() if p.is_file()}
    csv = {p.name: sha256(p) for p in sorted(OLIST.glob("*.csv"))}
    frame = build_dataset()
    # Physically separate August from all model, calibrator and threshold selection.
    development = frame.loc[frame.order_purchase_timestamp.lt(FINAL[3])]
    folds = [development_windows(development, cuts) for cuts in DEV]
    result = {
        "started_at_utc": now(), "features": FEATURES, "derived": DERIVED, "variants": VARIANTS,
        "compact_hgb": {"max_iter": 150, "max_leaf_nodes": 15, "min_samples_leaf": 80,
                        "l2_regularization": 2., "learning_rate": .07},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "source_hashes": csv, "prior_model_hashes": previous,
        "code_sha256": {name: sha256(ROOT / name) for name in
                        ("ml/improve_delay.py", "ml/evaluate_delay.py", "ml/train_delay.py", "ml/common.py")},
        "development_windows": [fold[3] for fold in folds], "development": {},
        "selection_rule": "Highest macro raw AP; lowest macro Brier calibration for winner; "
                          "highest pooled F1 under 10% pooled alerts, ties precision and higher cutoff.",
    }
    with threadpool_limits(limits=1):
        dev_scores = {}
        for name in VARIANTS:
            dev_scores[name] = {method: [] for method in CALIBRATIONS}
            for train, cal, evaluation, _ in folds:
                base = fit_base(name, train)
                for method in CALIBRATIONS:
                    model = calibrate(base, method, cal)
                    dev_scores[name][method].append(scores(model, evaluation))
            result["development"][name] = {}
            for method, predictions in dev_scores[name].items():
                rows = [metrics(fold[2].retrasado, p) for fold, p in zip(folds, predictions)]
                result["development"][name][method] = {"folds_at_0_5": rows, "macro": aggregate(rows)}
            print(f"Completado desarrollo: {name}", flush=True)

        summary = result["development"]
        name = max(VARIANTS, key=lambda key: summary[key]["raw"]["macro"]["average_precision"])
        method = min(CALIBRATIONS, key=lambda key: summary[name][key]["macro"]["brier"])
        y_dev = np.concatenate([fold[2].retrasado.to_numpy() for fold in folds])
        p_dev = np.concatenate(dev_scores[name][method])
        threshold = choose_threshold(y_dev, p_dev)
        result["development_selected_at_threshold"] = {
            "pooled": metrics(y_dev, p_dev, threshold),
            "folds": [metrics(fold[2].retrasado, p, threshold)
                      for fold, p in zip(folds, dev_scores[name][method])],
        }
        train, train_audit = window(development, None, FINAL[0], FINAL[1])
        cal, cal_audit = window(development, FINAL[1], FINAL[2], FINAL[3])
        model = calibrate(fit_base(name, train), method, cal)
        serialized = io.BytesIO()
        joblib.dump(model, serialized, compress=3)
        frozen_bytes = serialized.getvalue()
        round_trip = joblib.load(io.BytesIO(frozen_bytes))
        sample = pd.concat([cal.head(3), train.loc[train.peso_g.isna()].head(2),
                            train.loc[train.n_estados_vendedor.gt(1)].head(2)])
        np.testing.assert_array_equal(scores(model, cal), scores(round_trip, cal))
        with tempfile.TemporaryDirectory() as directory:
            check_file = Path(directory) / "candidate.joblib"
            check_file.write_bytes(frozen_bytes)
            result["compatibility"] = check_compatibility(model, check_file, sample)
        result["frozen"] = {"at_utc": now(), "variant": name, "calibration": method, "threshold": threshold,
                            "max_pooled_dev_alert_fraction": ALERT_BUDGET, "fit": train_audit, "calibration_window": cal_audit}
        print(json.dumps(result["frozen"]), flush=True)

        test, audit = window(frame, FINAL[3], FINAL[4], FINAL[5])
        old = joblib.load(OLD)
        y = test.retrasado.to_numpy()
        previous_p, new_p = scores(old, test), scores(round_trip, test)
        old_metrics = metrics(y, previous_p, OLD_THRESHOLD)
        if old_metrics["confusion_matrix_tn_fp_fn_tp"] != [[5127, 831], [279, 114]]:
            raise ValueError("No se reprodujo la comparación histórica sobre agosto")
        result["retrospective_august"] = {
            "scored_at_utc": now(), "window": audit, "old": old_metrics,
            "new": metrics(y, new_p, threshold),
            "constant_fit_prior": metrics(y, np.full(len(test), train.retrasado.mean())),
            "old_reliability": reliability(y, previous_p), "new_reliability": reliability(y, new_p),
            "paired_uncertainty": paired_day_bootstrap(test, new_p, previous_p, threshold),
            "subgroups": {},
        }
        for group, mask in {
            "same_state": test.mismo_estado.eq(1), "different_state": test.mismo_estado.eq(0),
            "delivery_gt_30d": test.dias_entrega.gt(30),
        }.items():
            result["retrospective_august"]["subgroups"][group] = {
                "old": metrics(y[mask], previous_p[mask], OLD_THRESHOLD),
                "new": metrics(y[mask], new_p[mask], threshold),
            }

    if previous != {p.name: sha256(p) for p in MODELS.iterdir() if p.is_file()} or \
       csv != {p.name: sha256(p) for p in sorted(OLIST.glob("*.csv"))}:
        raise ValueError("Cambió una entrada o artefacto previo")
    import hashlib
    result["artifact"] = {"name": artifact.name, "sha256": hashlib.sha256(frozen_bytes).hexdigest(),
                          "bytes": len(frozen_bytes), "activated": False}
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    with artifact.open("xb") as handle:
        handle.write(frozen_bytes)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "mejora_retrasos_metricas.json")
    parser.add_argument("--artifact", type=Path, default=MODELS / "delay_model_mejora_experimental.joblib")
    args = parser.parse_args()
    run(args.output, args.artifact)
