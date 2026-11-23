"""Pruebas de cálculo experimental. No ajustan estimadores."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml.evaluate_delay import metrics, reliability, run, select_threshold, undo_class_weights


class EvaluationTests(unittest.TestCase):
    def test_positive_class_metrics_and_matrix_order(self):
        result = metrics([0, 0, 1, 1], [.1, .6, .4, .9])
        self.assertEqual(result["confusion_matrix_tn_fp_fn_tp"], [[1, 1], [1, 1]])
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["recall"], .5)
        self.assertEqual(result["f1"], .5)
        self.assertAlmostEqual(result["roc_auc"], .75)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertAlmostEqual(result["brier"], .185)

    def test_constant_prior_baseline_ap_equals_prevalence_not_accuracy(self):
        result = metrics([0, 0, 0, 1], [.25] * 4)
        self.assertEqual(result["average_precision"], .25)
        self.assertEqual(result["roc_auc"], .5)
        self.assertEqual(result["recall"], 0.)
        self.assertEqual(result["confusion_matrix_tn_fp_fn_tp"], [[3, 0], [1, 0]])

    def test_no_invented_auc_for_single_class_or_empty_subgroup(self):
        self.assertIsNone(metrics([0, 0], [.1, .2])["roc_auc"])
        self.assertIsNone(metrics([0, 0], [.1, .2])["average_precision"])
        self.assertEqual(metrics([], [])["n"], 0)

    def test_invalid_probability_is_rejected(self):
        for values in ([np.nan, .1], [-.1, .5], [1.1, .5]):
            with self.assertRaises(ValueError):
                metrics([0, 1], values)

    def test_threshold_selection_uses_validation_f1_and_deterministic_tie(self):
        threshold, rows = select_threshold([0, 1], [.1, .9])
        self.assertEqual(len(rows), 99)
        self.assertEqual(threshold, .9)
        self.assertEqual(max(row["f1"] for row in rows), 1.)

    def test_weight_correction_is_only_an_algebraic_diagnostic(self):
        p = undo_class_weights([0., .5, 1.], .075)
        np.testing.assert_allclose(p, [0., .075, 1.])

    def test_reliability_accounts_for_zero_and_one_and_empty_bins(self):
        result = reliability([0, 1], [0., 1.])
        self.assertEqual(sum(row["n"] for row in result["bins"]), 2)
        self.assertEqual(result["ece_10_uniform_bins"], 0.)
        self.assertEqual(result["bins"][-1]["n"], 1)
        self.assertIsNone(result["bins"][5]["observed_rate"])

    def test_existing_artifact_is_never_overwritten_or_retrained(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "existing.joblib"
            path.write_bytes(b"existing artifact sentinel")
            with patch("ml.evaluate_delay.build_dataset", side_effect=AssertionError("No entrenar")):
                with self.assertRaises(FileExistsError):
                    run(path, Path(temporary) / "metrics.json")
            self.assertEqual(path.read_bytes(), b"existing artifact sentinel")


if __name__ == "__main__":
    unittest.main()
