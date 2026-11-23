"""Casos controlados en memoria; ningún test ajusta HGB ni K-Means."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from ml.common import delivered_orders
from ml.temporal import chronological_split
from ml.train_delay import CAT, NUM, build_dataset, build_pipeline
from ml.train_sellers import seller_metrics, seller_order_metrics

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.schemas import DelayInput
from app.services import delay_service


def orders_fixture():
    return pd.DataFrame([
        dict(order_id="a", customer_id="ca", order_status="delivered",
             order_purchase_timestamp="2018-01-01 12:00:00",
             order_estimated_delivery_date="2018-01-03 00:00:00",
             order_delivered_customer_date="2018-01-03 12:00:00"),
        dict(order_id="b", customer_id="cb", order_status="delivered",
             order_purchase_timestamp="2018-01-01 12:00:00",
             order_estimated_delivery_date="2018-01-03 00:00:00",
             order_delivered_customer_date="2018-01-05 12:00:00"),
    ])


def frames():
    return {
        "orders": orders_fixture(),
        "items": pd.DataFrame([
            ["a", 1, "p1", "s1", 10., 2.],
            ["a", 2, "p1", "s1", 10., 2.],
            ["a", 3, "p1", "s1", 10., 2.],
            ["b", 1, "p2", "s1", 20., 3.],
            ["b", 2, "p1", "s2", 5., 1.],
        ], columns=["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"]),
        "products": pd.DataFrame([["p1", 100.], ["p2", 200.]], columns=["product_id", "product_weight_g"]),
        "sellers": pd.DataFrame([["s1", "SP"], ["s2", "RJ"]], columns=["seller_id", "seller_state"]),
        "customers": pd.DataFrame([["ca", "ua", "SP"], ["cb", "ub", "RJ"]], columns=["customer_id", "customer_unique_id", "customer_state"]),
    }


def reviews_fixture():
    return pd.DataFrame([
        ["r1", "a", 1], ["r2", "a", 5], ["r3", "b", 5],
    ], columns=["review_id", "order_id", "review_score"])


class CalendarTargetTests(unittest.TestCase):
    def test_same_promised_day_is_not_late(self):
        frame = orders_fixture()
        frame.loc[0, "order_delivered_customer_date"] = "2018-01-03 23:59:59"
        self.assertEqual(delivered_orders(frame).set_index("order_id").loc["a", "retrasado"], 0)

    def test_next_calendar_day_is_late(self):
        frame = orders_fixture()
        frame.loc[0, "order_delivered_customer_date"] = "2018-01-04 00:00:01"
        self.assertEqual(delivered_orders(frame).set_index("order_id").loc["a", "retrasado"], 1)

    def test_elapsed_delivery_days_keep_hours(self):
        frame = orders_fixture()
        frame.loc[0, "order_delivered_customer_date"] = "2018-01-03 18:00:00"
        self.assertEqual(delivered_orders(frame).set_index("order_id").loc["a", "dias_entrega"], 2.25)

    def test_missing_target_is_excluded_and_counted_not_imputed(self):
        frame = orders_fixture()
        frame.loc[0, "order_delivered_customer_date"] = None
        prepared = delivered_orders(frame)
        self.assertEqual(list(prepared.order_id), ["b"])
        self.assertEqual(prepared.attrs["audit"]["missing_target_dates"], 1)

    def test_nondelivered_is_not_treated_as_punctual(self):
        frame = orders_fixture()
        frame.loc[0, "order_status"] = "shipped"
        self.assertEqual(list(delivered_orders(frame).order_id), ["b"])

    def test_impossible_target_dates_fail_instead_of_silent_deletion(self):
        frame = orders_fixture()
        frame.loc[0, "order_delivered_customer_date"] = "2017-12-31 12:00:00"
        with self.assertRaises(ValueError):
            delivered_orders(frame)


class DelayPreparationTests(unittest.TestCase):
    def test_multiple_items_sum_values_and_count_actual_units(self):
        result = build_dataset(**frames()).set_index("order_id")
        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc["a", "precio"], 30.)
        self.assertEqual(result.loc["a", "flete"], 6.)
        self.assertEqual(result.loc["a", "peso_g"], 300.)
        self.assertEqual(result.loc["a", "n_items"], 3)

    def test_calendar_promised_days_do_not_truncate_purchase_hour(self):
        self.assertTrue(build_dataset(**frames())["dias_estimados"].eq(2).all())

    def test_missing_weight_is_not_converted_to_zero(self):
        data = frames()
        data["products"].loc[0, "product_weight_g"] = np.nan
        result = build_dataset(**data).set_index("order_id")
        self.assertEqual(len(result), 2)
        self.assertTrue(np.isnan(result.loc["a", "peso_g"]))
        # El peso parcial conocido de b no debe aparentar ser su total.
        self.assertTrue(np.isnan(result.loc["b", "peso_g"]))

    def test_explicit_zero_weight_is_unknown(self):
        data = frames()
        data["products"].loc[0, "product_weight_g"] = 0
        self.assertTrue(build_dataset(**data)["peso_g"].isna().all())

    def test_dominant_state_preserves_all_multiseller_items(self):
        result = build_dataset(**frames()).set_index("order_id")
        self.assertEqual(result.loc["b", "estado_vendedor"], "SP")
        self.assertEqual(result.loc["b", "n_estados_vendedor"], 2)
        self.assertEqual(result.loc["b", "precio"], 25.)
        self.assertEqual(result.loc["b", "peso_g"], 300.)

    def test_state_tie_uses_count_then_alphabetical_state(self):
        data = frames()
        # b: SP y RJ con el mismo importe y cantidad -> RJ.
        data["items"].loc[data["items"].order_id.eq("b"), "price"] = 10.
        result = build_dataset(**data).set_index("order_id")
        self.assertEqual(result.loc["b", "estado_vendedor"], "RJ")
        # Añadir un segundo artículo SP sin cambiar su valor total -> SP.
        data["items"].loc[(data["items"].order_id.eq("b")) & data["items"].seller_id.eq("s1"), "price"] = 5.
        extra = pd.DataFrame([["b", 3, "p1", "s1", 5., 1.]], columns=data["items"].columns)
        data["items"] = pd.concat([data["items"], extra], ignore_index=True)
        self.assertEqual(build_dataset(**data).set_index("order_id").loc["b", "estado_vendedor"], "SP")

    def test_input_row_order_does_not_choose_origin(self):
        data = frames()
        expected = build_dataset(**data).sort_values("order_id")[NUM + CAT].reset_index(drop=True)
        shuffled = {key: frame.sample(frac=1, random_state=7) for key, frame in data.items()}
        actual = build_dataset(**shuffled).sort_values("order_id")[NUM + CAT].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)

    def test_joins_fail_for_duplicate_reference_key(self):
        data = frames()
        data["products"] = pd.concat([data["products"], data["products"].iloc[[0]]])
        with self.assertRaises(ValueError):
            build_dataset(**data)

    def test_joins_fail_for_orphan_product_or_order(self):
        for column in ("product_id", "order_id"):
            with self.subTest(column=column):
                data = frames()
                data["items"].loc[0, column] = "orphan"
                with self.assertRaises(ValueError):
                    build_dataset(**data)

    def test_duplicate_item_key_fails_instead_of_double_revenue(self):
        data = frames()
        data["items"] = pd.concat([data["items"], data["items"].iloc[[0]]])
        with self.assertRaises(ValueError):
            build_dataset(**data)

    def test_preprocessing_uses_only_train_and_never_metadata(self):
        train = build_dataset(**frames())
        train["estado_cliente"] = "SP"
        train["estado_vendedor"] = "SP"
        train.loc[0, "peso_g"] = np.nan
        prep = build_pipeline().named_steps["prep"]
        # Solo se ajusta el codificador, no un modelo de Machine Learning.
        prep.fit(train)
        test = train.iloc[[0]].copy()
        test["estado_cliente"] = "RJ"
        transformed = prep.transform(test)
        self.assertEqual(transformed.shape[1], len(NUM) + 2)
        self.assertTrue(np.isnan(transformed).any())  # HGB maneja el NaN nativamente.
        self.assertNotIn("RJ", prep.named_transformers_["cat"].categories_[0])
        self.assertFalse(any("order_id" in name or "timestamp" in name or "retrasado" in name for name in prep.get_feature_names_out()))
        classifier = build_pipeline().named_steps["clf"]
        self.assertFalse(classifier.early_stopping)
        self.assertEqual(classifier.class_weight, "balanced")

    def test_service_zero_weight_matches_training_policy_without_contract_change(self):
        model = Mock()
        model.predict_proba.return_value = np.array([[.8, .2]])
        payload = DelayInput(precio=10, flete=2, peso_g=0, n_items=1, dias_estimados=2,
                             estado_cliente="sp", estado_vendedor="rj", mes=1, dia_semana=0)
        with patch.object(delay_service, "_get_model", return_value=model):
            response = delay_service.predict(payload)
        row = model.predict_proba.call_args.args[0]
        self.assertTrue(np.isnan(row.loc[0, "peso_g"]))
        self.assertEqual(row.loc[0, "estado_cliente"], "SP")
        self.assertEqual(set(row.columns), set(NUM + CAT))
        self.assertEqual(response.probabilidad_retraso, .2)
        self.assertTrue(response.modelo_entrenado)


class SellerPreparationTests(unittest.TestCase):
    def pairs(self, reviews=None):
        data = frames()
        return seller_order_metrics(data["orders"], data["items"], reviews_fixture() if reviews is None else reviews)

    def test_multiple_items_and_reviews_do_not_multiply_pairs(self):
        pairs = self.pairs()
        self.assertEqual(len(pairs), 3)
        self.assertFalse(pairs.duplicated(["seller_id", "order_id"]).any())
        self.assertEqual(pairs.attrs["audit"]["eligible_item_rows"], 5)
        self.assertEqual(pairs.attrs["audit"]["rows_after_reviews_join"], 3)
        a = pairs.set_index(["seller_id", "order_id"]).loc[("s1", "a")]
        self.assertEqual(a["nota"], 3.)
        self.assertEqual(a["facturacion"], 30.)

    def test_means_are_per_seller_order_and_revenue_is_per_item(self):
        data = frames()
        with patch("ml.train_sellers.MIN_PEDIDOS", 1):
            result = seller_metrics(data["orders"], data["items"], reviews_fixture())
        self.assertEqual(result.loc["s1", "n_pedidos"], 2)
        self.assertEqual(result.loc["s1", "facturacion"], 50.)
        self.assertEqual(result.loc["s1", "nota_media"], 4.)
        self.assertEqual(result.loc["s1", "tasa_retraso"], .5)
        self.assertEqual(result.loc["s1", "dias_entrega_medio"], 3.)
        self.assertEqual(result.loc["s2", "facturacion"], 5.)
        self.assertEqual(result.loc["s2", "n_pedidos"], 1)

    def test_missing_reviews_are_not_imputed_as_satisfaction(self):
        data = frames()
        with patch("ml.train_sellers.MIN_PEDIDOS", 1):
            result = seller_metrics(data["orders"], data["items"], reviews_fixture().iloc[:2])
        self.assertEqual(result.loc["s1", "nota_media"], 3.)
        self.assertNotIn("s2", result.index)
        self.assertEqual(result.attrs["audit"]["sellers_missing_metrics_after_minimum"], 1)

    def test_shared_review_id_across_different_orders_is_allowed(self):
        reviews = reviews_fixture()
        reviews.loc[2, "review_id"] = "r1"
        self.assertEqual(len(self.pairs(reviews)), 3)

    def test_duplicate_review_order_pair_fails(self):
        reviews = reviews_fixture()
        with self.assertRaises(ValueError):
            self.pairs(pd.concat([reviews, reviews.iloc[[0]]]))

    def test_review_row_order_does_not_change_metrics(self):
        expected = self.pairs().sort_values(["seller_id", "order_id"]).reset_index(drop=True)
        actual = self.pairs(reviews_fixture().sample(frac=1, random_state=2)).sort_values(["seller_id", "order_id"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual)


class TemporalSplitTests(unittest.TestCase):
    def fixture(self):
        rows = [
            ("train", "2018-05-01", "2018-05-04"),
            ("train_pending", "2018-05-31", "2018-06-01"),
            ("validation", "2018-06-01", "2018-06-04"),
            ("validation_pending", "2018-06-30", "2018-07-01"),
            ("test", "2018-07-01", "2018-07-03"),
            ("test_pending", "2018-07-31", "2018-09-01"),
            ("outside", "2018-08-01", "2018-08-02"),
        ]
        frame = pd.DataFrame(rows, columns=["order_id", "order_purchase_timestamp", "label_available_at"])
        frame["retrasado"] = [0, 1, 0, 1, 1, 1, 0]
        return frame

    def split(self, frame=None):
        return chronological_split(
            self.fixture() if frame is None else frame, train_end="2018-06-01",
            validation_start="2018-06-01", validation_end="2018-07-01",
            test_start="2018-07-01", test_end="2018-08-01", observation_end="2018-09-01",
        )

    def test_labels_at_or_after_cutoff_are_purged_and_accounted(self):
        split = self.split()
        self.assertEqual(list(split.train.order_id), ["train"])
        self.assertEqual(list(split.validation.order_id), ["validation"])
        self.assertEqual(list(split.test.order_id), ["test"])
        self.assertEqual(set(split.excluded.exclusion_reason), {
            "train_label_not_available", "validation_label_not_available",
            "test_label_not_available", "outside_purchase_windows",
        })

    def test_sets_are_disjoint_and_all_rows_are_accounted(self):
        split = self.split()
        parts = [split.train, split.validation, split.test, split.excluded]
        ids = [set(part.order_id) for part in parts]
        self.assertEqual(sum(len(part) for part in parts), len(self.fixture()))
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                self.assertFalse(ids[a] & ids[b])
        self.assertLess(split.train.order_purchase_timestamp.max(), split.validation.order_purchase_timestamp.min())
        self.assertLess(split.validation.order_purchase_timestamp.max(), split.test.order_purchase_timestamp.min())

    def test_no_future_results_in_train_or_validation(self):
        split = self.split()
        self.assertTrue(split.train.label_available_at.lt("2018-06-01").all())
        self.assertTrue(split.validation.label_available_at.lt("2018-07-01").all())
        self.assertTrue(split.test.label_available_at.lt("2018-09-01").all())

    def test_maturation_gaps_are_explicit_not_silently_dropped(self):
        split = chronological_split(
            self.fixture(), train_end="2018-05-15", validation_start="2018-06-01",
            validation_end="2018-06-15", test_start="2018-07-01",
            test_end="2018-08-01", observation_end="2018-09-01",
        )
        reasons = split.excluded.set_index("order_id")["exclusion_reason"]
        self.assertEqual(reasons.loc["train_pending"], "train_label_maturation_gap")
        self.assertEqual(reasons.loc["validation_pending"], "validation_label_maturation_gap")
        self.assertEqual(sum(len(part) for part in [split.train, split.validation, split.test, split.excluded]), 7)

    def test_shuffle_does_not_change_temporal_selection(self):
        a, b = self.split(), self.split(self.fixture().sample(frac=1, random_state=1))
        for name in ("train", "validation", "test"):
            self.assertEqual(list(getattr(a, name).order_id), list(getattr(b, name).order_id))

    def test_invalid_cuts_and_duplicate_orders_fail(self):
        with self.assertRaises(ValueError):
            chronological_split(self.fixture(), validation_start="2018-09-01", test_start="2018-08-01")
        with self.assertRaises(ValueError):
            self.split(pd.concat([self.fixture(), self.fixture().iloc[[0]]]))

    def test_missing_or_pre_purchase_label_fails(self):
        for value in (None, "2018-01-01"):
            frame = self.fixture()
            frame.loc[0, "label_available_at"] = value
            with self.assertRaises(ValueError):
                self.split(frame)


if __name__ == "__main__":
    unittest.main()
