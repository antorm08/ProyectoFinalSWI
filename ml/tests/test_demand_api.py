"""API de demanda y artefacto: contrato, banda y caída a modo provisional.

Ningún test optimiza un SARIMAX. Los artefactos se construyen con `smooth()`
sobre parámetros fijados a mano, que aplica el filtro de Kalman sin ajustar nada.
"""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import olist
from app.main import app
from app.services import demand_model as dm
from app.services import demand_service

RUTA = "/api/demand/forecast"


async def _get(path):
    eventos = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(evento):
        eventos.append(evento)

    camino, _, consulta = path.partition("?")
    await app({
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": "GET", "scheme": "http", "path": camino, "raw_path": camino.encode(),
        "root_path": "", "query_string": consulta.encode(), "headers": [],
        "client": ("127.0.0.1", 1234), "server": ("test", 80),
    }, receive, send)
    estado = next(e["status"] for e in eventos if e["type"] == "http.response.start")
    cuerpo = b"".join(e.get("body", b"") for e in eventos if e["type"] == "http.response.body")
    return estado, json.loads(cuerpo or b"null")


def get(path):
    return asyncio.run(_get(path))


def serie_demo(dias=240, inicio="2018-01-01"):
    paso = np.arange(dias)
    valores = 150 + 20 * np.sin(2 * np.pi * paso / 7) + 0.2 * paso
    return pd.Series(valores, index=pd.date_range(inicio, periods=dias, freq="D"), name="pedidos")


def parametros(modelo) -> np.ndarray:
    """Parámetros plausibles por nombre, sin pasar por el optimizador."""
    valores = []
    for nombre in modelo.param_names:
        if nombre == "sigma2":
            valores.append(0.01)
        elif nombre.startswith(("ar.", "ma.")):
            valores.append(0.3)
        else:
            valores.append(0.0)
    return np.array(valores, dtype=float)


def bundle_demo(serie=None, spec=None, **extra):
    serie = serie_demo() if serie is None else serie
    spec = spec or dm.Especificacion(order=(1, 0, 0), seasonal_order=(0, 0, 0, 7))
    modelo = dm.construir(serie, spec)
    resultados = modelo.smooth(parametros(modelo))
    return dm.empaquetar(serie, spec, resultados, **extra)


class Artefacto(unittest.TestCase):
    def test_la_ida_y_vuelta_reproduce_el_pronostico_guardado(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "demand_model.joblib"
            joblib.dump(bundle_demo(), ruta)
            comprobacion = dm.verificar(joblib.load(ruta))
            self.assertLessEqual(max(comprobacion["desvio_maximo"].values()), 1e-6)

    def test_el_artefacto_no_arrastra_el_objeto_de_statsmodels(self):
        bundle = bundle_demo()
        self.assertEqual(sorted(bundle["spec"]), ["exog", "order", "seasonal_order", "transform"])
        self.assertIsInstance(bundle["params"], np.ndarray)
        self.assertNotIn("results", bundle)

    def test_rechaza_el_formato_antiguo(self):
        antiguo = {"results": object(), "history": serie_demo()}
        with self.assertRaises(dm.BundleIncompatible):
            dm.desempaquetar(antiguo)

    def test_rechaza_columnas_exogenas_inesperadas(self):
        bundle = bundle_demo()
        bundle["exog_columns"] = ["otra_cosa"]
        with self.assertRaises(dm.BundleIncompatible):
            dm.desempaquetar(bundle)

    def test_rechaza_un_historico_con_huecos(self):
        serie = serie_demo()
        bundle = bundle_demo(serie=serie)
        bundle["history"] = serie.drop(serie.index[100])
        with self.assertRaises(dm.BundleIncompatible):
            dm.desempaquetar(bundle)

    def test_la_banda_nunca_baja_de_cero_ni_se_invierte(self):
        serie = serie_demo()
        spec = dm.Especificacion(order=(1, 0, 0), seasonal_order=(0, 0, 0, 7))
        modelo = dm.construir(serie, spec)
        marco = dm.pronosticar(modelo.smooth(parametros(modelo)), serie, spec, 60, 0.95)
        self.assertTrue((marco["inferior"] >= 0).all())
        self.assertTrue((marco["inferior"] <= marco["valor"]).all())
        self.assertTrue((marco["valor"] <= marco["superior"]).all())


class ApiSinModelo(unittest.TestCase):
    """Sin artefacto servible la API responde igual: 200 y modo provisional."""

    def setUp(self):
        self.carpeta = tempfile.TemporaryDirectory()
        self.ruta = Path(self.carpeta.name) / "demand_model.joblib"
        self.parche = patch.object(demand_service, "DEMAND_MODEL", self.ruta)
        self.parche.start()
        demand_service._cache = None

    def tearDown(self):
        self.parche.stop()
        demand_service._cache = None
        self.carpeta.cleanup()

    def test_sin_artefacto_responde_provisional(self):
        estado, cuerpo = get(f"{RUTA}?horizonte=14&historico=30")
        self.assertEqual(estado, 200)
        self.assertFalse(cuerpo["modelo_entrenado"])
        self.assertEqual(cuerpo["modelo"], "provisional")
        self.assertIsNone(cuerpo["metricas"])
        futuros = [p for p in cuerpo["serie"] if p["tipo"] == "pronostico"]
        self.assertEqual(len(futuros), 14)
        self.assertTrue(all(p["inferior"] is None for p in futuros))

    def test_un_artefacto_corrupto_no_produce_un_500(self):
        self.ruta.write_bytes(b"esto no es un joblib")
        estado, cuerpo = get(f"{RUTA}?horizonte=7")
        self.assertEqual(estado, 200)
        self.assertFalse(cuerpo["modelo_entrenado"])

    def test_un_artefacto_con_esquema_viejo_cae_a_provisional(self):
        joblib.dump({"results": "lo que sea", "history": serie_demo()}, self.ruta)
        estado, cuerpo = get(f"{RUTA}?horizonte=7")
        self.assertEqual(estado, 200)
        self.assertFalse(cuerpo["modelo_entrenado"])

    def test_valida_los_limites_del_horizonte_y_del_nivel(self):
        self.assertEqual(get(f"{RUTA}?horizonte=200")[0], 422)
        self.assertEqual(get(f"{RUTA}?horizonte=0")[0], 422)
        self.assertEqual(get(f"{RUTA}?nivel=1.5")[0], 422)
        self.assertEqual(get(f"{RUTA}?nivel=0.2")[0], 422)
        self.assertEqual(get(f"{RUTA}?horizonte=90")[0], 200)


class ApiConModelo(unittest.TestCase):
    def setUp(self):
        self.carpeta = tempfile.TemporaryDirectory()
        self.ruta = Path(self.carpeta.name) / "demand_model.joblib"
        medidas = {"mae": 12.5, "mase": 0.61, "rel_mae": 0.78, "cobertura_80": 0.82,
                   "baseline_mae": 16.0, "periodo": "2018-07-01 a 2018-07-28"}
        joblib.dump(
            bundle_demo(metrics=medidas, window={"inicio": "2018-01-01", "fin": "2018-08-28"}),
            self.ruta,
        )
        self.parche = patch.object(demand_service, "DEMAND_MODEL", self.ruta)
        self.parche.start()
        demand_service._cache = None

    def tearDown(self):
        self.parche.stop()
        demand_service._cache = None
        self.carpeta.cleanup()

    def test_publica_la_banda_las_metricas_y_el_ancla(self):
        estado, cuerpo = get(f"{RUTA}?horizonte=21&historico=40&nivel=0.8")
        self.assertEqual(estado, 200)
        self.assertTrue(cuerpo["modelo_entrenado"])
        self.assertIn("SARIMA", cuerpo["modelo"])
        self.assertEqual(cuerpo["nivel_confianza"], 0.8)
        self.assertEqual(cuerpo["metricas"]["mae"], 12.5)
        self.assertEqual(cuerpo["ventana"]["fin"], "2018-08-28")
        historicos = [p for p in cuerpo["serie"] if p["tipo"] == "historico"]
        futuros = [p for p in cuerpo["serie"] if p["tipo"] == "pronostico"]
        self.assertEqual((len(historicos), len(futuros)), (40, 21))
        self.assertEqual(cuerpo["ancla"], historicos[-1]["fecha"])
        self.assertEqual(
            futuros[0]["fecha"],
            str((pd.Timestamp(cuerpo["ancla"]) + pd.Timedelta(days=1)).date()),
        )

    def test_la_banda_contiene_al_pronostico_y_no_es_negativa(self):
        _, cuerpo = get(f"{RUTA}?horizonte=30&nivel=0.95")
        for punto in [p for p in cuerpo["serie"] if p["tipo"] == "pronostico"]:
            self.assertLessEqual(punto["inferior"], punto["valor"])
            self.assertLessEqual(punto["valor"], punto["superior"])
            self.assertGreaterEqual(punto["inferior"], 0)

    def test_un_nivel_mas_alto_ensancha_la_banda(self):
        _, estrecha = get(f"{RUTA}?horizonte=10&nivel=0.8")
        _, ancha = get(f"{RUTA}?horizonte=10&nivel=0.95")
        def anchura(cuerpo):
            futuros = [p for p in cuerpo["serie"] if p["tipo"] == "pronostico"]
            return sum(p["superior"] - p["inferior"] for p in futuros)
        self.assertGreater(anchura(ancha), anchura(estrecha))

    def test_los_puntos_historicos_no_llevan_banda(self):
        _, cuerpo = get(f"{RUTA}?horizonte=7&historico=15")
        for punto in [p for p in cuerpo["serie"] if p["tipo"] == "historico"]:
            self.assertIsNone(punto["inferior"])
            self.assertIsNone(punto["superior"])

    def test_la_cache_se_invalida_al_cambiar_el_artefacto(self):
        _, antes = get(f"{RUTA}?horizonte=7")
        self.assertTrue(antes["modelo_entrenado"])
        self.ruta.write_bytes(b"artefacto roto")
        _, despues = get(f"{RUTA}?horizonte=7")
        self.assertFalse(despues["modelo_entrenado"])


class DatosPosterioresAlArtefacto(unittest.TestCase):
    """El histórico no se queda congelado en el artefacto si hay días nuevos."""

    def setUp(self):
        self.serie = serie_demo()
        self.fin = self.serie.index[-1]
        self.carpeta = tempfile.TemporaryDirectory()
        self.ruta = Path(self.carpeta.name) / "demand_model.joblib"
        joblib.dump(bundle_demo(serie=self.serie), self.ruta)
        self.parche = patch.object(demand_service, "DEMAND_MODEL", self.ruta)
        self.parche.start()
        demand_service._cache = None

    def tearDown(self):
        self.parche.stop()
        demand_service._cache = None
        self.carpeta.cleanup()

    def pedidos(self, fechas, por_dia=200):
        """Tabla de pedidos que, agregada por día, reproduce `fechas`."""
        filas = [{"order_purchase_timestamp": f"{dia:%Y-%m-%d} 10:00:00"}
                 for dia, n in fechas.items() for _ in range(int(n))]
        return pd.DataFrame(filas)

    def con_datos(self, tabla):
        return patch.multiple(
            olist, available=lambda nombre: True, load=lambda nombre: tabla
        )

    def test_anexa_los_dias_nuevos_sin_reajustar(self):
        nuevos = pd.date_range(self.fin + pd.Timedelta(days=1), periods=5, freq="D")
        completo = pd.concat([self.serie, pd.Series(210.0, index=nuevos)])
        with self.con_datos(self.pedidos(completo.round().astype(int))):
            _, cuerpo = get(f"{RUTA}?horizonte=7&historico=10")
        self.assertTrue(cuerpo["modelo_entrenado"])
        self.assertEqual(cuerpo["ancla"], str(nuevos[-1].date()))
        futuros = [p for p in cuerpo["serie"] if p["tipo"] == "pronostico"]
        self.assertEqual(futuros[0]["fecha"], str((nuevos[-1] + pd.Timedelta(days=1)).date()))
        self.assertIsNotNone(futuros[0]["inferior"])

    def test_no_anexa_si_la_continuacion_tiene_un_hueco(self):
        nuevos = pd.date_range(self.fin + pd.Timedelta(days=2), periods=3, freq="D")  # salta un día
        completo = pd.concat([self.serie, pd.Series(210.0, index=nuevos)])
        with self.con_datos(self.pedidos(completo.round().astype(int))):
            _, cuerpo = get(f"{RUTA}?horizonte=7&historico=10")
        self.assertEqual(cuerpo["ancla"], str(self.fin.date()))

    def test_no_anexa_un_dia_sin_pedidos(self):
        nuevos = pd.date_range(self.fin + pd.Timedelta(days=1), periods=3, freq="D")
        valores = pd.Series([210.0, 0.0, 205.0], index=nuevos)
        completo = pd.concat([self.serie, valores])
        with self.con_datos(self.pedidos(completo.round().astype(int))):
            _, cuerpo = get(f"{RUTA}?horizonte=7&historico=10")
        self.assertEqual(cuerpo["ancla"], str(self.fin.date()))

    def test_sin_csv_disponibles_se_queda_en_el_historico_del_artefacto(self):
        with patch.object(olist, "available", lambda nombre: False):
            _, cuerpo = get(f"{RUTA}?horizonte=7&historico=10")
        self.assertTrue(cuerpo["modelo_entrenado"])
        self.assertEqual(cuerpo["ancla"], str(self.fin.date()))


if __name__ == "__main__":
    unittest.main()
