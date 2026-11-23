"""Validation, artifact integrity and regression/classification API integration."""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.main import app
from app.schemas import DeliveryDurationInput
from app.services import delivery_duration_service as eta

ORDER = {"precio": 120, "flete": 20, "peso_g": 1500, "n_items": 1,
         "estado_cliente": "RJ", "estado_vendedor": "SP"}
EXAMPLE = {**ORDER, "fecha_compra": "2018-08-06"}
NEW_ROUTE = "/api/delay/predict-duration"


async def post(path, body):
    """Invoke the real FastAPI router/middleware via ASGI, without a server dependency."""
    events = []

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    async def send(event):
        events.append(event)

    await app({
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
        "root_path": "", "query_string": b"", "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234), "server": ("test", 80),
    }, receive, send)
    status = next(e["status"] for e in events if e["type"] == "http.response.start")
    return status, json.loads(b"".join(e.get("body", b"") for e in events if e["type"] == "http.response.body"))


class DeliveryDurationTests(unittest.TestCase):
    def test_exact_features_and_unknown_weight(self):
        order = DeliveryDurationInput(**{**EXAMPLE, "estado_cliente": " sp ", "peso_g": 0})
        row = eta.model_row(order)
        expected = {"precio", "flete", "peso_g", "n_items", "mes", "dia_semana",
                    "mismo_estado", "estado_cliente", "estado_vendedor"}
        self.assertEqual(set(row.columns), expected)
        self.assertTrue(np.isnan(row.peso_g.iloc[0]))
        self.assertEqual((row.mes.iloc[0], row.dia_semana.iloc[0], row.mismo_estado.iloc[0]), (8, 0, 1))

    def test_nonfinite_monetary_inputs_are_invalid(self):
        for value in (float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                DeliveryDurationInput(**{**EXAMPLE, "precio": value})
        status, body = asyncio.run(post(NEW_ROUTE, {**EXAMPLE, "precio": float("inf")}))
        self.assertEqual(status, 422)
        self.assertNotIn("input", body["detail"][0])

    def test_missing_artifact_is_unavailable_without_heuristic(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(eta, "DELIVERY_DURATION_MODEL", Path(temp) / "missing.joblib"):
                status, body = asyncio.run(post(NEW_ROUTE, EXAMPLE))
        self.assertEqual(status, 503)
        self.assertIn("no disponible", body["detail"])
        self.assertNotIn("dias_entrega_estimados", body)

    def test_untrusted_payload_is_never_deserialized_even_if_size_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "untrusted.joblib"
            path.write_bytes(b"x" * eta.MODEL_SIZE)
            with patch.object(eta, "DELIVERY_DURATION_MODEL", path), patch.object(eta.joblib, "load") as load:
                with self.assertLogs(eta.logger, "WARNING"):
                    status, _ = asyncio.run(post(NEW_ROUTE, EXAMPLE))
                load.assert_not_called()
        self.assertEqual(status, 503)

    def test_validations_cannot_be_bypassed_via_legacy_fields(self):
        for body in (ORDER, {**ORDER, "mes": 8}, {**EXAMPLE, "dias_estimados": 100},
                     {**EXAMPLE, "mes": 8}, {**EXAMPLE, "fecha_compra": "2018-08-06T00:00:00"},
                     {**EXAMPLE, "fecha_compra": 20180806}, {**EXAMPLE, "fecha_compra": "2018-02-30"},
                     {**EXAMPLE, "estado_vendedor": "XX"}, {**EXAMPLE, "n_items": 0}):
            with self.subTest(body=body), patch.object(eta, "predict") as predict:
                status, _ = asyncio.run(post(NEW_ROUTE, body))
                self.assertEqual(status, 422)
                predict.assert_not_called()

    @unittest.skipUnless(eta.DELIVERY_DURATION_MODEL.is_file(), "Se requiere el artefacto local aprobado")
    def test_real_artifact_infers_without_promise_with_and_without_date(self):
        with threadpool_limits(limits=2):
            status, dated = asyncio.run(post(NEW_ROUTE, EXAMPLE))
            no_date_status, undated = asyncio.run(post(NEW_ROUTE, {**ORDER, "mes": 8, "dia_semana": 0}))
        self.assertEqual((status, no_date_status), (200, 200))
        self.assertAlmostEqual(dated["dias_entrega_estimados"], 10.90887913, places=7)
        self.assertEqual(undated["dias_entrega_estimados"], dated["dias_entrega_estimados"])
        self.assertEqual(dated["fecha_llegada_estimada"], "2018-08-16")
        self.assertIsNone(undated["fecha_llegada_estimada"])
        self.assertEqual(dated["modelo"], eta.MODEL_NAME)
        self.assertTrue(dated["experimental"])
        self.assertNotIn("probabilidad_retraso", dated)

    def test_legacy_classifier_has_independent_request_and_response(self):
        status, body = asyncio.run(post("/api/delay/predict", {**ORDER, "mes": 8, "dia_semana": 0, "dias_estimados": 20}))
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {"probabilidad_retraso", "nivel_riesgo", "modelo_entrenado"})
