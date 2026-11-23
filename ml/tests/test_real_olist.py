"""Integración sobre CSV originales: preparación únicamente, sin entrenamiento."""
import contextlib
import hashlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml.common import MODELS, OLIST, delivered_orders, load
from ml.temporal import chronological_split
from ml import train_delay, train_sellers

HASHES = {
    "orders": "8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf",
    "customers": "983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2",
    "order_items": "0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279",
    "products": "3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc",
    "sellers": "1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e",
    "order_reviews": "012b61c7593e34f51fa614efdf802b9c7056ce6aae5307ddb93236e7cfc797d7",
}


def artifacts():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in MODELS.glob("*") if path.is_file()}


@unittest.skipUnless(all((OLIST / f"olist_{name}_dataset.csv").exists() for name in HASHES), "Requiere los seis CSV de Olist")
class RealOlistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial_artifacts = artifacts()
        cls.raw = load("orders")
        cls.items = load("order_items")
        cls.delivered = delivered_orders(cls.raw)
        cls.delay = train_delay.build_dataset()
        cls.split = chronological_split(cls.delay)
        cls.pairs = train_sellers.seller_order_metrics()
        cls.sellers = train_sellers.seller_metrics()

    @classmethod
    def tearDownClass(cls):
        if artifacts() != cls.initial_artifacts:
            raise AssertionError("Se modificaron artefactos durante las pruebas")

    def test_original_csv_hashes_match_validation(self):
        for name, expected in HASHES.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((OLIST / f"olist_{name}_dataset.csv").read_bytes()).hexdigest(), expected)

    def test_target_changes_from_7826_to_6534_over_same_96470(self):
        self.assertEqual(len(self.delivered), 96470)
        old = self.delivered.order_delivered_customer_date.gt(self.delivered.order_estimated_delivery_date)
        self.assertEqual(int(old.sum()), 7826)
        self.assertEqual(int(self.delivered.retrasado.sum()), 6534)
        self.assertEqual(int((old & self.delivered.retrasado.eq(0)).sum()), 1292)
        self.assertEqual(int(self.delay.retrasado.sum()), 6534)

    def test_feature_joins_preserve_orders_and_unknown_weights(self):
        self.assertEqual(len(self.delay), 96470)
        self.assertEqual(self.delay.order_id.nunique(), 96470)
        self.assertEqual(int(self.delay.peso_g.isna().sum()), 22)
        self.assertEqual(int(self.delay.peso_g.eq(0).sum()), 0)
        self.assertEqual(int(self.delay.n_estados_vendedor.gt(1).sum()), 509)
        for column in train_delay.NUM:
            self.assertFalse(np.isinf(self.delay[column]).any(), column)
            if column != "peso_g":
                self.assertFalse(self.delay[column].isna().any(), column)
        self.assertTrue(self.delay.dias_estimados.ge(0).all())

    def test_prices_and_items_are_conserved_across_aggregations(self):
        items = self.items.loc[self.items.order_id.isin(self.delivered.order_id)]
        self.assertEqual(len(items), 110189)
        self.assertEqual(int(self.delay.n_items.sum()), len(items))
        self.assertAlmostEqual(self.delay.precio.sum(), items.price.sum(), places=6)
        self.assertAlmostEqual(self.delay.flete.sum(), items.freight_value.sum(), places=6)
        self.assertAlmostEqual(self.pairs.facturacion.sum(), items.price.sum(), places=6)

    def test_seller_order_join_and_metrics_are_unweighted(self):
        self.assertEqual(len(self.pairs), 97811)
        self.assertFalse(self.pairs.duplicated(["seller_id", "order_id"]).any())
        self.assertEqual(self.pairs.attrs["audit"]["rows_after_reviews_join"], 97811)
        self.assertEqual(len(self.sellers), 1766)
        self.assertEqual(self.pairs.seller_id.nunique(), 2970)
        # Comparación independiente: medias por pareja, no por ítem.
        expected = self.pairs.groupby("seller_id").agg(
            n_pedidos=("order_id", "size"), facturacion=("facturacion", "sum"),
            nota_media=("nota", "mean"), tasa_retraso=("retrasado", "mean"),
            dias_entrega_medio=("dias_entrega", "mean"),
        )
        expected = expected.loc[expected.n_pedidos.ge(5)].dropna()
        pd.testing.assert_frame_equal(self.sellers, expected)
        expected_labels = self.delivered.set_index("order_id").retrasado
        self.assertTrue(self.pairs.retrasado.eq(self.pairs.order_id.map(expected_labels)).all())

    def test_real_split_has_no_temporal_or_label_leakage(self):
        parts = [self.split.train, self.split.validation, self.split.test, self.split.excluded]
        self.assertEqual(sum(map(len, parts)), len(self.delay))
        combined = pd.concat(parts)
        self.assertFalse(combined.order_id.duplicated().any())
        self.assertEqual(set(combined.order_id), set(self.delay.order_id))
        self.assertTrue(self.split.train.order_purchase_timestamp.lt("2018-05-01").all())
        self.assertTrue(self.split.train.label_available_at.lt("2018-06-01").all())
        self.assertTrue(self.split.validation.order_purchase_timestamp.ge("2018-06-01").all())
        self.assertTrue(self.split.validation.order_purchase_timestamp.lt("2018-07-01").all())
        self.assertTrue(self.split.validation.label_available_at.lt("2018-08-01").all())
        self.assertTrue(self.split.test.order_purchase_timestamp.ge("2018-08-01").all())
        self.assertTrue(self.split.test.order_purchase_timestamp.lt("2018-09-01").all())
        self.assertTrue(self.split.test.label_available_at.lt("2018-10-18").all())
        for part in parts[:3]:
            self.assertEqual(part.retrasado.nunique(), 2)

    def test_real_split_counts_are_reproducible(self):
        self.assertEqual([len(self.split.train), len(self.split.validation), len(self.split.test)], [71023, 6070, 6351])
        self.assertEqual(self.split.excluded.exclusion_reason.value_counts().to_dict(), {
            "train_label_maturation_gap": 6749, "validation_label_maturation_gap": 6156,
            "train_label_not_available": 95, "validation_label_not_available": 26,
        })

    def test_real_multistate_policy_is_independent_of_row_order(self):
        shuffled = train_delay.build_dataset(
            orders=self.raw.sample(frac=1, random_state=42),
            items=self.items.sample(frac=1, random_state=43),
        )
        expected = self.delay.set_index("order_id").sort_index()
        actual = shuffled.set_index("order_id").sort_index()
        pd.testing.assert_series_equal(expected.estado_vendedor, actual.estado_vendedor)
        pd.testing.assert_series_equal(expected.n_estados_vendedor, actual.n_estados_vendedor)
        np.testing.assert_allclose(expected.peso_g, actual.peso_g, equal_nan=True)
        np.testing.assert_allclose(expected.precio, actual.precio)

    def test_check_data_cannot_fit_models_or_write_artifacts(self):
        with patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("No entrenar")), \
             patch("sklearn.cluster.KMeans.fit", side_effect=AssertionError("No entrenar")), \
             patch("sklearn.ensemble.HistGradientBoostingClassifier.fit", side_effect=AssertionError("No entrenar")), \
             patch("joblib.dump", side_effect=AssertionError("No escribir modelos")), \
             patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("No escribir tablas")), \
             contextlib.redirect_stdout(io.StringIO()):
            delay_summary = train_delay.main(check_only=True)
            seller_summary = train_sellers.main(check_only=True)
        self.assertEqual(delay_summary["preparation"]["late_calendar_day"], 6534)
        self.assertEqual(seller_summary["eligible_sellers"], 1766)
        self.assertEqual(artifacts(), self.initial_artifacts)


if __name__ == "__main__":
    unittest.main()
