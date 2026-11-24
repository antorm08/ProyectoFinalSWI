"""Casos controlados del componente 2 (series temporales); nada de esto ajusta SARIMAX.

Las series sintéticas reproducen a propósito las tres patologías que
`docs/validacion_olist.md` sección 5.2 mide en Olist: un hueco de cobertura al
principio, una cola truncada al final y un pico irrepetible de Black Friday.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml.demand_series import (
    EXOG_COLUMNAS, PERIODO, coverage, coverage_window, daily_orders, final_holdout,
    mase, metrics, rel_mae, rolling_origin, seasonal_naive, weekday_mean,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.services.demand_calendar import CalendarioFueraDeRango, calendar_exog


def pedidos(fechas):
    """Tabla mínima de pedidos: un identificador único por fila."""
    return pd.DataFrame(
        [{"order_id": f"o{n:05d}", "order_purchase_timestamp": f} for n, f in enumerate(fechas)]
    )


def serie_sintetica(dias=600, inicio="2017-01-01", nivel=150.0, semilla=0):
    """Tendencia suave, estacionalidad semanal y ruido; siempre positiva."""
    rng = np.random.default_rng(semilla)
    t = np.arange(dias)
    valores = nivel + 0.25 * t + 30 * np.sin(2 * np.pi * t / PERIODO) + rng.normal(0, 8, dias)
    indice = pd.date_range(inicio, periods=dias, freq="D")
    serie = pd.Series(np.clip(valores, 1, None), index=indice, name="pedidos")
    serie.attrs["audit"] = {"source_orders": int(serie.sum())}
    return serie


class SerieDiaria(unittest.TestCase):
    def test_cuenta_por_dia_y_rellena_los_dias_sin_registros(self):
        serie = daily_orders(pedidos([
            "2018-01-01 10:00:00", "2018-01-01 23:59:59", "2018-01-01 00:00:00",
            "2018-01-03 12:00:00",
        ]))
        self.assertEqual(list(serie), [3.0, 0.0, 1.0])
        self.assertEqual(serie.index.freqstr, "D")
        self.assertEqual(serie.attrs["audit"]["days_without_records"], 1)
        self.assertEqual(serie.attrs["audit"]["orders_in_series"], 4)

    def test_cualquier_estado_cuenta_como_demanda(self):
        tabla = pedidos(["2018-01-01 10:00:00", "2018-01-02 10:00:00"])
        tabla["order_status"] = ["canceled", "delivered"]
        self.assertEqual(list(daily_orders(tabla)), [1.0, 1.0])

    def test_rechaza_pedidos_repetidos(self):
        tabla = pedidos(["2018-01-01 10:00:00", "2018-01-02 10:00:00"])
        tabla["order_id"] = "mismo"
        with self.assertRaises(ValueError):
            daily_orders(tabla)

    def test_rechaza_fecha_de_compra_ausente(self):
        tabla = pedidos(["2018-01-01 10:00:00", "2018-01-02 10:00:00"])
        tabla.loc[1, "order_purchase_timestamp"] = None
        with self.assertRaises(ValueError):
            daily_orders(tabla)


class VentanaDeCobertura(unittest.TestCase):
    def test_recorta_el_hueco_inicial_y_conserva_la_rampa(self):
        serie = serie_sintetica()
        serie.iloc[:4] = 0.0                      # los ceros de enero de 2017
        ventana = coverage_window(serie)
        self.assertEqual(str(ventana.inicio.date()), "2017-01-05")
        self.assertEqual(ventana.audit["coverage_gaps"][0]["dias"], 4)
        self.assertEqual(ventana.audit["window"]["ceros"], 0)

    def test_recorta_la_cola_truncada(self):
        serie = serie_sintetica()
        serie.iloc[-30:] = 2.0                    # septiembre y octubre de 2018
        ventana = coverage_window(serie)
        self.assertIsNotNone(ventana.audit["first_truncated_day"])
        self.assertLess(ventana.fin, serie.index[-30] + pd.Timedelta(days=PERIODO))
        self.assertGreater(ventana.audit["discarded"]["dias_despues"], 20)

    def test_descarta_el_descenso_final_completo_no_solo_su_fondo(self):
        """El archivo se agota poco a poco: en Olist cae 16 días hasta cero.

        Un umbral absoluto encuentra el final del descenso, no su comienzo, y
        dejaría dentro días ya infrarregistrados que el modelo leería como una
        caída de la demanda. Lo que se exige no es acertar la fecha exacta de
        inicio, sino que no sobreviva ningún día claramente infrarregistrado.
        """
        serie = serie_sintetica()
        nivel = float(serie.iloc[-40:-16].mean())
        serie.iloc[-16:] = np.linspace(0.85, 0.0, 16) * nivel
        ventana = coverage_window(serie)
        self.assertIsNotNone(ventana.audit["final_descent_start"])
        self.assertLess(ventana.fin, serie.index[-1])
        # Ningún día de la ventana queda por debajo de la mitad del nivel previo.
        self.assertGreater(float(ventana.serie.iloc[-7:].min()), 0.5 * nivel)
        self.assertGreater(ventana.audit["discarded"]["dias_despues"], 10)

    def test_la_marcha_atras_no_recorta_un_valle_interior(self):
        serie = serie_sintetica()
        serie.iloc[200:210] = serie.iloc[200:210] * 0.35       # bajón interior, se recupera
        ventana = coverage_window(serie)
        self.assertIsNone(ventana.audit["final_descent_start"])
        self.assertEqual(len(ventana.serie), len(serie))

    def test_un_dia_suelto_a_cero_no_es_un_hueco(self):
        serie = serie_sintetica()
        serie.iloc[300] = 0.0
        ventana = coverage_window(serie)
        self.assertEqual(ventana.audit["coverage_gaps"], [])
        self.assertEqual(len(ventana.serie), len(serie))

    def test_la_rampa_de_crecimiento_no_se_confunde_con_truncamiento(self):
        serie = serie_sintetica(nivel=20.0)       # arranca bajo y crece 10x
        ventana = coverage_window(serie)
        self.assertIsNone(ventana.audit["first_truncated_day"])

    def test_la_auditoria_compara_con_la_ventana_heredada(self):
        ventana = coverage_window(serie_sintetica())
        heredada = ventana.audit["legacy_window"]
        self.assertEqual(heredada["inicio"], "2017-01-01")
        self.assertGreater(heredada["pedidos"], 0)

    def test_exige_frecuencia_diaria_declarada(self):
        serie = serie_sintetica()
        suelta = pd.Series(serie.to_numpy(), index=pd.DatetimeIndex(serie.index.to_numpy()))
        with self.assertRaises(ValueError):
            coverage_window(suelta)

    def test_falla_si_la_ventana_resultante_es_demasiado_corta(self):
        with self.assertRaises(ValueError):
            coverage_window(serie_sintetica(dias=100))


class Baselines(unittest.TestCase):
    def setUp(self):
        self.train = serie_sintetica(dias=200)

    def test_el_naive_estacional_repite_la_ultima_semana(self):
        pred = seasonal_naive(self.train, 10)
        esperado = list(self.train.iloc[-PERIODO:]) + list(self.train.iloc[-PERIODO:-PERIODO + 3])
        np.testing.assert_allclose(pred.to_numpy(), esperado)
        self.assertEqual(pred.index[0], self.train.index[-1] + pd.Timedelta(days=1))

    def test_la_media_por_dia_de_semana_usa_las_ultimas_cuatro_semanas(self):
        pred = weekday_mean(self.train, 7, semanas=4)
        reciente = self.train.iloc[-28:]
        for fecha, valor in pred.items():
            esperado = reciente[reciente.index.day_of_week == fecha.day_of_week].mean()
            self.assertAlmostEqual(valor, esperado)

    def test_exigen_historia_suficiente(self):
        with self.assertRaises(ValueError):
            seasonal_naive(self.train.iloc[:3], 7)
        with self.assertRaises(ValueError):
            weekday_mean(self.train.iloc[:20], 7, semanas=4)


class OrigenMovil(unittest.TestCase):
    def setUp(self):
        self.serie = serie_sintetica(dias=600)

    def test_ningun_origen_ve_datos_posteriores_a_su_corte(self):
        for train, test in rolling_origin(self.serie):
            self.assertLess(train.index.max(), test.index.min())
            self.assertEqual(test.index.min(), train.index.max() + pd.Timedelta(days=1))

    def test_el_holdout_queda_fuera_de_todos_los_origenes(self):
        reservado = self.serie.index[-28:]
        for _, test in rolling_origin(self.serie, holdout=28):
            self.assertTrue(test.index.intersection(reservado).empty)

    def test_los_origenes_llegan_en_orden_y_la_ventana_es_expansiva(self):
        particiones = rolling_origin(self.serie, n_origenes=5, horizonte=28, paso=14)
        self.assertEqual(len(particiones), 5)
        largos = [len(train) for train, _ in particiones]
        self.assertEqual(largos, sorted(largos))
        for anterior, siguiente in zip(particiones[:-1], particiones[1:]):
            self.assertLess(anterior[1].index.min(), siguiente[1].index.min())

    def test_el_holdout_final_es_el_tramo_reservado(self):
        train, test = final_holdout(self.serie, holdout=28)
        self.assertEqual(len(test), 28)
        self.assertEqual(train.index.max() + pd.Timedelta(days=1), test.index.min())

    def test_falla_cuando_la_serie_no_da_para_un_origen(self):
        with self.assertRaises(ValueError):
            rolling_origin(self.serie.iloc[:40])


class Metricas(unittest.TestCase):
    def setUp(self):
        self.train = serie_sintetica(dias=200)
        self.test = serie_sintetica(dias=28, inicio=str((self.train.index[-1] + pd.Timedelta(days=1)).date()))

    def test_mae_rmse_y_sesgo_con_valores_conocidos(self):
        y = pd.Series([10.0, 20.0, 30.0], index=pd.date_range("2018-01-01", periods=3, freq="D"))
        pred = pd.Series([12.0, 18.0, 33.0], index=y.index)
        resultado = metrics(y, pred)
        self.assertAlmostEqual(resultado["mae"], (2 + 2 + 3) / 3)
        self.assertAlmostEqual(resultado["rmse"], float(np.sqrt((4 + 4 + 9) / 3)))
        self.assertAlmostEqual(resultado["bias"], (2 - 2 + 3) / 3)

    def test_el_mae_relativo_del_naive_estacional_vale_exactamente_uno(self):
        base = seasonal_naive(self.train, len(self.test))
        base.index = self.test.index
        self.assertEqual(rel_mae(self.test, base, self.train), 1.0)

    def test_el_mase_escala_con_el_naive_en_muestra(self):
        train = pd.Series(
            np.arange(100, dtype=float), index=pd.date_range("2018-01-01", periods=100, freq="D")
        )
        y = pd.Series([100.0, 101.0], index=pd.date_range("2018-04-11", periods=2, freq="D"))
        pred = pd.Series([107.0, 108.0], index=y.index)
        self.assertAlmostEqual(mase(y, pred, train), 7.0 / PERIODO)

    def test_las_metricas_exigen_alineacion_temporal(self):
        pred = seasonal_naive(self.train, len(self.test))
        pred.index = pred.index + pd.Timedelta(days=1)
        with self.assertRaises(ValueError):
            metrics(self.test, pred)          # mismo largo, fechas desplazadas un día
        with self.assertRaises(ValueError):
            metrics(self.test, pred.iloc[:-1])

    def test_la_cobertura_cuenta_los_dias_dentro_de_la_banda(self):
        y = pd.Series([10.0, 20.0, 30.0, 40.0], index=pd.date_range("2018-01-01", periods=4, freq="D"))
        self.assertEqual(coverage(y, y - 1, y + 1), 1.0)
        self.assertEqual(coverage(y, y + 5, y + 10), 0.0)
        mixta = coverage(y, pd.Series([9.0, 25.0, 29.0, 45.0], index=y.index),
                         pd.Series([11.0, 30.0, 31.0, 50.0], index=y.index))
        self.assertEqual(mixta, 0.5)

    def test_rechaza_una_banda_invertida(self):
        y = pd.Series([10.0, 20.0], index=pd.date_range("2018-01-01", periods=2, freq="D"))
        with self.assertRaises(ValueError):
            coverage(y, y + 1, y - 1)


class Calendario(unittest.TestCase):
    def test_marca_el_black_friday_y_su_cola(self):
        exog = calendar_exog(pd.date_range("2017-11-22", "2017-11-30", freq="D"))
        self.assertEqual(list(exog.columns), list(EXOG_COLUMNAS))
        self.assertEqual(exog.loc["2017-11-24", "black_friday"], 1.0)
        self.assertEqual(exog["black_friday_cola"].sum(), 4.0)
        self.assertEqual(exog["black_friday"].sum(), 1.0)

    def test_sirve_para_fechas_futuras_y_es_determinista(self):
        futuro = pd.date_range("2018-11-20", periods=10, freq="D")
        primera, segunda = calendar_exog(futuro), calendar_exog(futuro)
        pd.testing.assert_frame_equal(primera, segunda)
        self.assertEqual(primera.loc["2018-11-23", "black_friday"], 1.0)

    def test_marca_las_fiestas_fijas_y_las_moviles(self):
        exog = calendar_exog(pd.date_range("2018-01-01", "2018-12-31", freq="D"))
        self.assertEqual(exog.loc["2018-12-25", "festivo"], 1.0)
        self.assertEqual(exog.loc["2018-02-12", "festivo"], 1.0)   # Carnaval
        self.assertEqual(exog.loc["2018-03-30", "festivo"], 1.0)   # Viernes Santo
        self.assertEqual(exog.loc["2018-07-04", "festivo"], 0.0)

    def test_falla_fuera_de_los_años_declarados_en_vez_de_devolver_ceros(self):
        with self.assertRaises(CalendarioFueraDeRango):
            calendar_exog(pd.date_range("2021-01-01", periods=3, freq="D"))

    def test_exige_un_indice_de_fechas(self):
        with self.assertRaises(TypeError):
            calendar_exog(pd.Index([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
