"""Banco de algoritmos: contrato de cada método y honestidad del protocolo.

Se prueban a fondo los métodos baratos y deterministas —los que el informe
recomienda— y el andamiaje (evaluación, bootstrap y sensibilidad) con un método
falso. Ningún test ajusta un SARIMAX ni entrena un modelo.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from ml import benchmark_demand as B
from ml.demand_series import PERIODO, rolling_origin

BARATOS = ("naive_estacional", "media_dia_semana_4s", "media_dia_semana_8s",
           "mediana_dia_semana_8s", "ewma_dia_semana", "perfil_por_nivel",
           "perfil_por_nivel_amort", "combinacion_horizonte")


def serie(dias=420, inicio="2017-06-01", semilla=0):
    generador = np.random.default_rng(semilla)
    paso = np.arange(dias)
    valores = 200 + 0.2 * paso + 30 * np.sin(2 * np.pi * paso / PERIODO) + generador.normal(0, 8, dias)
    indice = pd.date_range(inicio, periods=dias, freq="D")
    return pd.Series(np.clip(valores, 1, None), index=indice, name="pedidos")


class ContratoDeLosMetodos(unittest.TestCase):
    def setUp(self):
        self.train = serie()

    def test_devuelven_el_numero_de_dias_pedido_y_las_fechas_siguientes(self):
        for nombre in BARATOS:
            with self.subTest(nombre):
                pred = B.CATALOGO[nombre][0](self.train, 28)
                self.assertEqual(len(pred), 28)
                self.assertEqual(pred.index[0], self.train.index[-1] + pd.Timedelta(days=1))
                self.assertEqual(pred.index.freqstr, "D")

    def test_no_devuelven_valores_imposibles(self):
        for nombre in BARATOS:
            with self.subTest(nombre):
                valores = np.asarray(B.CATALOGO[nombre][0](self.train, 60), dtype=float)
                self.assertTrue(np.isfinite(valores).all())
                self.assertTrue((valores >= 0).all())

    def test_ninguno_mira_mas_alla_del_origen(self):
        """Truncar la serie no puede cambiar un pronóstico que ya se emitió."""
        corto = self.train.iloc[:-30]
        for nombre in BARATOS:
            with self.subTest(nombre):
                a = B.CATALOGO[nombre][0](corto, 14)
                b = B.CATALOGO[nombre][0](pd.concat([corto, self.train.iloc[-30:]]).iloc[:len(corto)], 14)
                np.testing.assert_allclose(a.to_numpy(), b.to_numpy())

    def test_el_perfil_reproduce_una_serie_puramente_semanal(self):
        patron = [100.0, 120.0, 140.0, 130.0, 160.0, 90.0, 70.0]
        limpia = pd.Series(patron * 60, index=pd.date_range("2018-01-01", periods=420, freq="D"))
        for nombre in ("media_dia_semana_4s", "media_dia_semana_8s", "perfil_por_nivel",
                       "ewma_dia_semana"):
            with self.subTest(nombre):
                pred = B.CATALOGO[nombre][0](limpia, PERIODO)
                esperado = [patron[f.day_of_week] for f in pred.index]
                np.testing.assert_allclose(pred.to_numpy(), esperado, rtol=1e-6)

    def test_la_combinacion_por_horizonte_cambia_de_metodo_en_el_dia_ocho(self):
        pred = B.combinacion_por_horizonte(self.train, 21)
        naive = B.CATALOGO["naive_estacional"][0](self.train, 21)
        perfil = B.perfil_por_nivel(self.train, 21)
        np.testing.assert_allclose(pred.to_numpy()[:PERIODO], naive.to_numpy()[:PERIODO])
        np.testing.assert_allclose(pred.to_numpy()[PERIODO:], perfil.to_numpy()[PERIODO:])

    def test_el_perfil_por_nivel_separa_forma_y_nivel(self):
        """Doblar el nivel reciente dobla el pronóstico, sin tocar la forma."""
        doblada = self.train.copy()
        doblada.iloc[-28:] = doblada.iloc[-28:] * 2
        base = B.perfil_por_nivel(self.train, 14).to_numpy()
        subida = B.perfil_por_nivel(doblada, 14).to_numpy()
        self.assertGreater(subida.mean(), base.mean() * 1.5)


class Protocolo(unittest.TestCase):
    def setUp(self):
        self.particiones = rolling_origin(serie(500), n_origenes=4, horizonte=28,
                                          paso=14, holdout=28)

    def test_el_baseline_obtiene_exactamente_uno(self):
        datos = B.evaluar("naive", B.CATALOGO["naive_estacional"][0], self.particiones)
        self.assertAlmostEqual(datos["rel_mae"], 1.0, places=12)
        self.assertEqual(datos["gana_al_baseline"], 0)

    def test_el_resumen_trae_dispersion_y_no_solo_la_media(self):
        datos = B.evaluar("perfil", B.perfil_por_nivel, self.particiones)
        for clave in ("rel_mae", "rel_mae_mediana", "rel_mae_sd", "rel_mae_min",
                      "rel_mae_max", "gana_al_baseline", "mae_por_semana", "bias_por_semana"):
            self.assertIn(clave, datos)
        self.assertEqual(len(datos["mae_por_semana"]), 4)

    def test_el_bootstrap_no_ve_diferencia_entre_un_metodo_y_si_mismo(self):
        errores = {}
        B.evaluar("a", B.perfil_por_nivel, self.particiones, errores)
        B.evaluar("b", B.perfil_por_nivel, self.particiones, errores)
        fila = B.bootstrap_pareado(errores, (("a", "b"),), len(self.particiones))[0]
        self.assertEqual(fila["delta_mae"], 0.0)
        self.assertFalse(fila["significativo"])

    def test_el_bootstrap_detecta_una_mejora_grande(self):
        errores = {}
        B.evaluar("bueno", B.perfil_por_nivel, self.particiones, errores)
        B.evaluar("malo", lambda t, s: B.perfil_por_nivel(t, s) * 0.5, self.particiones, errores)
        fila = B.bootstrap_pareado(errores, (("bueno", "malo"),), len(self.particiones))[0]
        self.assertLess(fila["delta_mae"], 0)
        self.assertTrue(fila["significativo"])
        self.assertGreater(fila["p_es_mejor"], 0.9)

    def test_el_bootstrap_remuestrea_origenes_y_es_reproducible(self):
        errores = {}
        B.evaluar("a", B.perfil_por_nivel, self.particiones, errores)
        B.evaluar("b", B.CATALOGO["naive_estacional"][0], self.particiones, errores)
        uno = B.bootstrap_pareado(errores, (("a", "b"),), len(self.particiones))[0]
        otro = B.bootstrap_pareado(errores, (("a", "b"),), len(self.particiones))[0]
        self.assertEqual(uno, otro)
        self.assertEqual(uno["referencia"], "b")

    def test_la_sensibilidad_recorre_las_tres_rejillas(self):
        salida = B.sensibilidad(self.particiones)
        self.assertEqual(sorted(salida), ["ewma_dia_semana", "media_dia_semana", "perfil_por_nivel"])
        self.assertEqual(salida["perfil_por_nivel"]["total"], 25)
        for datos in salida.values():
            self.assertLessEqual(datos["mejor"], datos["mediana"])
            self.assertLessEqual(datos["mediana"], datos["peor"])

    def test_el_holdout_queda_fuera_del_banco(self):
        reservado = serie(500).index[-28:]
        for _, test in self.particiones:
            self.assertTrue(test.index.intersection(reservado).empty)


if __name__ == "__main__":
    unittest.main()
