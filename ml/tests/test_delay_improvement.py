"""Checks for the second delay experiment, without using the final cohort."""
import io
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ml.improve_delay import ACTIVE, DEV, FINAL, build_variant, calibrate, choose_threshold, run, scores, window


class ImprovementTests(unittest.TestCase):
    def test_label_at_cutoff_and_purchase_limits_are_excluded(self):
        frame = pd.DataFrame({
            "order_id": list("abcde"),
            "order_purchase_timestamp": pd.to_datetime(["2018-01-01"] * 3 + ["2018-02-01", "2017-12-31"]),
            "label_available_at": pd.to_datetime(["2018-01-03", "2018-01-15", "2018-03-01", "2018-02-10", "2018-01-01"]),
            "retrasado": [0, 1, 1, 0, 1],
        })
        part, audit = window(frame, "2018-01-01", "2018-02-01", "2018-03-01")
        self.assertEqual(part.order_id.tolist(), ["a", "b"])
        self.assertEqual(audit["unavailable_n"], 1)
        self.assertEqual(audit["unavailable_late_retrospective"], 1)

    def test_all_development_label_deadlines_precede_august(self):
        for cuts in DEV:
            self.assertEqual(sorted(set(cuts)), list(cuts))
            self.assertLess(cuts[-1], FINAL[3])

    def test_alert_budget_does_not_split_tied_scores(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        p = np.array([.9, .8, .8, .5, .1, .1])
        self.assertEqual(choose_threshold(y, p, .34), .9)
        self.assertEqual(choose_threshold(y, p, .5), .8)
        self.assertGreater(choose_threshold(y, p, .1), 1.)

    def test_never_write_to_active_model_path(self):
        with self.assertRaises(FileExistsError):
            run(Path("unused.json"), ACTIVE)

    def test_derived_transformer_uses_api_fields_and_preserves_unknown_weight(self):
        frame = self.make_frame(40, 42)
        row = frame.iloc[[0]].copy()
        row["peso_g"] = np.nan
        transformer = build_variant("hgb_compact_weight3_derived").steps[0][1]
        transformed = transformer.transform(row)
        self.assertTrue(transformed.weight_per_item.isna().all())
        self.assertAlmostEqual(transformed.price_per_item.iloc[0], row.precio.iloc[0] / row.n_items.iloc[0])
        self.assertNotIn("price_per_item", row.columns)

    def test_calibration_keeps_base_predictions_and_survives_roundtrip(self):
        train, cal, validation = [self.make_frame(120, seed) for seed in (1, 2, 3)]
        with threadpool_limits(limits=1):
            base = build_variant("logistic").fit(train.drop(columns="retrasado"), train.retrasado)
            original = scores(base, validation)
            for method in ("sigmoid", "isotonic"):
                calibrated = calibrate(base, method, cal)
                np.testing.assert_array_equal(scores(base, validation), original)
                buffer = io.BytesIO()
                joblib.dump(calibrated, buffer)
                buffer.seek(0)
                np.testing.assert_array_equal(scores(calibrated, validation), scores(joblib.load(buffer), validation))

    @staticmethod
    def make_frame(n, seed):
        rng = np.random.default_rng(seed)
        price = rng.uniform(1, 300, n)
        return pd.DataFrame({
            "precio": price, "flete": rng.uniform(0, 100, n), "peso_g": rng.uniform(0, 1000, n),
            "n_items": rng.integers(1, 3, n), "dias_estimados": rng.integers(1, 30, n),
            "mes": rng.integers(1, 13, n), "dia_semana": rng.integers(0, 7, n), "mismo_estado": 0,
            "estado_cliente": "SP", "estado_vendedor": "RJ", "retrasado": (price > 180).astype(int),
        })
