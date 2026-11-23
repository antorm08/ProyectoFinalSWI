"""Focused checks for the offline duration experiment."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ml.compare_delivery_regression import (
    DEV, FEATURES, FINAL, TARGET, breakdown, build_model, late_diagnostic,
    macro, maturity_window, promised_baselines, regression_metrics, run, split_for_cut,
)


class DeliveryRegressionTests(unittest.TestCase):
    def test_label_at_cutoff_and_purchase_end_are_exclusive(self):
        frame = pd.DataFrame({
            "order_id": ["a", "b", "c", "d"],
            "order_purchase_timestamp": pd.to_datetime(["2018-02-01", "2018-02-02", "2018-03-01", "2018-01-31"]),
            "label_available_at": pd.to_datetime(["2018-02-05", "2018-04-01", "2018-03-05", "2018-02-05"]),
            TARGET: [4., 58., 4., 5.],
        })
        sample, audit = maturity_window(frame, "2018-02-01", "2018-03-01", "2018-04-01")
        self.assertEqual(sample.order_id.tolist(), ["a"])
        self.assertEqual(audit["unavailable_retrospective"]["n"], 1)
        self.assertEqual(audit["unavailable_retrospective"]["median"], 58.)

    def test_cuts_are_chronological_and_never_select_on_august(self):
        for cut in DEV:
            self.assertEqual(list(cut), sorted(set(cut)))
            self.assertLessEqual(cut[3], FINAL[1])
        with self.assertRaises(ValueError):
            split_for_cut(pd.DataFrame(), ("2018-02-01", "2018-01-01", "2018-03-01", "2018-04-01"))

    def test_metric_units_inclusive_tolerances_and_signed_bias(self):
        m = regression_metrics([2, 4, 10, 10], [0, 4, 15, 4])
        self.assertEqual(m["mae"], 3.25)
        self.assertEqual(m["medae"], 3.5)
        self.assertEqual(m["rmse"], np.sqrt(65 / 4))
        self.assertEqual(m["within_2_pct"], 50)
        self.assertEqual(m["within_5_pct"], 75)
        self.assertEqual(m["bias_pred_minus_real"], -.75)
        self.assertIsNone(regression_metrics([2.], [1.])["r2"])

    def test_duration_bands_do_not_drop_or_duplicate_boundaries_or_extremes(self):
        y = np.array([0., 5., 10., 15., 30., 209.])
        groups = breakdown(y, {"model": y})
        self.assertEqual([group["model"]["n"] for group in groups.values()], [2, 1, 1, 1, 1])

    def test_promise_retains_fractional_time_and_calendar_baseline(self):
        frame = pd.DataFrame({
            "order_purchase_timestamp": pd.to_datetime(["2018-02-01 18:00"]),
            "order_estimated_delivery_date": pd.to_datetime(["2018-02-03 00:00"]),
            "dias_estimados": [2],
            "retrasado": [1],
        })
        p = promised_baselines(frame)
        self.assertEqual(p["olist_timestamp"][0], 1.25)
        self.assertEqual(p["olist_calendar"][0], 2)
        self.assertEqual(late_diagnostic(frame, np.array([2.24]))["recall"], 0)
        self.assertEqual(late_diagnostic(frame, np.array([2.25]))["recall"], 1)

    def test_derivations_use_only_purchase_fields_and_preserve_missing_weight(self):
        frame = pd.DataFrame({
            "precio": [100., 100.], "flete": [20., 20.], "peso_g": [1000., np.nan], "n_items": [2, 2],
            "dias_estimados": [15, 30], "mes": [3, 4], "dia_semana": [1, 2], "mismo_estado": [0, 0],
            "estado_cliente": ["RJ", "PA"], "estado_vendedor": ["SP", "SP"],
        })[FEATURES]
        transform = build_model("hgb_derived").steps[0][1]
        result = transform.transform(frame)
        self.assertEqual(result.flete_unitario.tolist(), [10., 10.])
        self.assertEqual(result.misma_region.tolist(), [True, False])
        self.assertTrue(np.isnan(result.peso_unitario.iloc[1]))

    def test_pipeline_fit_ignores_metadata_and_categories_seen_only_at_inference(self):
        frame = pd.DataFrame({
            "precio": [1., 2., 3.], "flete": [1., 2., 3.], "peso_g": [1., np.nan, 3.], "n_items": [1, 2, 3],
            "dias_estimados": [10, 11, 12], "mes": [1, 2, 3], "dia_semana": [1, 2, 3], "mismo_estado": [0, 0, 1],
            "estado_cliente": ["SP", "SP", "RJ"], "estado_vendedor": ["PA", "PA", "RJ"],
            TARGET: [4., 5., 100.],
        })
        with threadpool_limits(limits=1):
            pipeline = build_model("ridge").fit(frame.iloc[:2], frame[TARGET].iloc[:2])
            self.assertTrue(np.isfinite(pipeline.predict(frame.iloc[2:])).all())
        prep = pipeline.named_steps["columntransformer"]
        self.assertEqual(prep.named_transformers_["cat"].categories_[0].tolist(), ["SP"])
        self.assertEqual(prep.named_transformers_["num"].named_steps["simpleimputer"].statistics_[2], 1.)
        self.assertNotIn(TARGET, prep.get_feature_names_out())
        no_promise = build_model("hgb_without_promise").named_steps["columntransformer"]
        self.assertNotIn("dias_estimados", no_promise.transformers[1][2])

    def test_macro_is_equal_weight_by_month(self):
        a = regression_metrics([1., 2.], [2., 3.])
        b = regression_metrics([1., 2., 3., 4.], [4., 5., 6., 7.])
        self.assertEqual(macro([{"x": a}, {"x": b}])["x"]["mae"], 2.)

    def test_active_artifact_path_is_rejected_before_training(self):
        with self.assertRaises(FileExistsError):
            run(Path("unused_delivery_results"), Path("backend/app/models/delay_model.joblib"))
