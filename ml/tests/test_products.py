"""Clasificación de productos por imagen: preprocesado, servicio, API y protocolo de entrenamiento.

La mayoría de las pruebas usa imágenes sintéticas y un modelo falso, sin depender de Fashion-MNIST
ni del artefacto entrenado. Las que necesitan esos recursos se omiten si no están disponibles.

Ejecutar desde la raíz:  python -m unittest ml.tests.test_products
"""
import asyncio
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.api.routes import products as route  # noqa: E402
from app.main import app  # noqa: E402
from app.services import product_service as ps  # noqa: E402
from ml import train_products as tp  # noqa: E402

FIXTURES = ROOT / "frontend" / "tests" / "fixtures"
EJEMPLOS = ROOT / "frontend" / "public" / "ejemplos"
FONDO_NORMALIZADO = (0 - ps.MEAN) / ps.STD     # valor de un píxel negro tras normalizar
HAS_MODEL = ps.PRODUCT_MODEL.is_file()
HAS_DATA = (tp.DATA_DIR / "FashionMNIST" / "raw" / "train-images-idx3-ubyte").is_file()


# ----------------------------------------------------------------------------- utilidades
def encode(img: Image.Image, fmt: str = "PNG", **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def prenda(size=(200, 200), box=(60, 40, 140, 160), fondo="white", color=(40, 40, 40), mode="RGB") -> Image.Image:
    """'Prenda' sintética: un polígono oscuro (camiseta esquemática) sobre fondo liso."""
    img = Image.new(mode, size, fondo)
    x0, y0, x1, y1 = box
    w = x1 - x0
    ImageDraw.Draw(img).polygon(
        [(x0, y0), (x1, y0), (x1, y0 + (y1 - y0) // 3), (x1 - w // 5, y0 + (y1 - y0) // 3),
         (x1 - w // 5, y1), (x0 + w // 5, y1), (x0 + w // 5, y0 + (y1 - y0) // 3), (x0, y0 + (y1 - y0) // 3)],
        fill=color)
    return img


def logits_para(probs) -> torch.Tensor:
    """Logits cuyo softmax da exactamente `probs`."""
    return torch.log(torch.tensor([probs], dtype=torch.float32))


def modelo_falso(probs):
    return lambda x: logits_para(probs).repeat(x.shape[0], 1)


def multipart(field: str, filename: str, data: bytes, content_type: str):
    boundary = "----pruebacommerceai"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


async def post_file(data: bytes, content_type="image/png", field="imagen", filename="foto.png"):
    """Invoca la app FastAPI real vía ASGI (router, validación y middleware), sin servidor."""
    body, ctype = multipart(field, filename, data, content_type)
    events = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(event):
        events.append(event)

    path = "/api/products/classify"
    await app({
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(), "root_path": "",
        "query_string": b"", "headers": [(b"content-type", ctype.encode()), (b"content-length", str(len(body)).encode())],
        "client": ("127.0.0.1", 1234), "server": ("test", 80),
    }, receive, send)
    status = next(e["status"] for e in events if e["type"] == "http.response.start")
    return status, json.loads(b"".join(e.get("body", b"") for e in events if e["type"] == "http.response.body"))


# ----------------------------------------------------------------------------- preprocesado
class PreprocessTests(unittest.TestCase):
    def test_formato_fashion_mnist(self):
        x = ps.preprocess(encode(prenda()))
        self.assertEqual(x.shape, (1, 1, 28, 28))
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(np.isfinite(x).all())
        # Las esquinas son fondo negro: tras normalizar valen (0 - media) / desviación.
        for esquina in (x[0, 0, 0, 0], x[0, 0, 0, -1], x[0, 0, -1, 0], x[0, 0, -1, -1]):
            self.assertAlmostEqual(float(esquina), FONDO_NORMALIZADO, places=4)
        self.assertGreater(x.max(), 1.0, "la prenda debe quedar clara sobre fondo negro")

    def test_fondo_claro_y_oscuro_dan_la_misma_entrada(self):
        claro = ps.preprocess(encode(prenda(fondo="white", color=(0, 0, 0))))
        oscuro = ps.preprocess(encode(prenda(fondo="black", color=(255, 255, 255))))
        np.testing.assert_allclose(claro, oscuro, atol=1e-5)

    def test_recorte_hace_invariante_el_encuadre(self):
        """La misma prenda llenando la foto o en un rincón de un lienzo grande produce la misma entrada."""
        llena = prenda(size=(105, 125), box=(2, 2, 102, 122))              # 2 px de margen
        lienzo = Image.new("RGB", (600, 600), "white")
        lienzo.paste(llena, (420, 30))                                     # ocupa ~1/6 del lado, en un rincón
        np.testing.assert_allclose(ps.preprocess(encode(llena)), ps.preprocess(encode(lienzo)), atol=1e-5)

    def test_conserva_proporcion_y_centra(self):
        alta = Image.new("L", (300, 300), 255)
        ImageDraw.Draw(alta).rectangle((140, 20, 160, 280), fill=0)        # prenda alta y estrecha
        x = ps.preprocess(encode(alta))[0, 0] > FONDO_NORMALIZADO + 0.5
        filas, cols = np.flatnonzero(x.any(1)), np.flatnonzero(x.any(0))
        self.assertGreaterEqual(len(filas), 27, "el lado mayor ocupa el alto completo")
        self.assertLess(len(cols), 6, "no se deforma a cuadrado")
        self.assertLessEqual(abs((cols[0] + cols[-1]) / 2 - 13.5), 1, "queda centrada")

    def test_corrige_orientacion_exif(self):
        foto = prenda(size=(200, 300), box=(50, 40, 150, 260))
        exif = Image.Exif()
        exif[0x0112] = 6                       # la cámara guardó la foto girada: mostrar rotando 270°
        girada = encode(foto.rotate(90, expand=True), "PNG", exif=exif)
        np.testing.assert_allclose(ps.preprocess(girada), ps.preprocess(encode(foto)), atol=1e-5)

    def test_transparencia_se_compone_sobre_blanco(self):
        transparente = prenda(fondo=(0, 0, 0, 0), color=(40, 40, 40, 255), mode="RGBA")
        sobre_blanco = prenda(fondo="white", color=(40, 40, 40))
        np.testing.assert_allclose(ps.preprocess(encode(transparente)), ps.preprocess(encode(sobre_blanco)), atol=1e-5)

    def test_fondo_gris_con_ruido_se_recorta_igual(self):
        limpia = ps.preprocess(encode(prenda(size=(400, 400), box=(100, 80, 260, 320), fondo=(215,) * 3)))
        ruido = np.random.default_rng(0).normal(0, 6, (400, 400, 1))
        con_ruido = np.asarray(prenda(size=(400, 400), box=(100, 80, 260, 320), fondo=(215,) * 3), np.float32) + ruido
        x = ps.preprocess(encode(Image.fromarray(np.clip(con_ruido, 0, 255).astype(np.uint8)), "JPEG"))
        self.assertLess(np.abs(limpia - x).mean(), 0.08)

    def test_imagen_lisa_no_tiene_prenda(self):
        for color in ("white", "black", (200, 200, 200)):
            with self.subTest(color=color), self.assertRaises(ps.NoObjectError):
                ps.preprocess(encode(Image.new("RGB", (100, 100), color)))

    def test_archivo_dañado_o_enorme_es_invalido(self):
        for data in (b"", b"no es una imagen", encode(prenda())[:60]):
            with self.subTest(data=data[:12]), self.assertRaises(ps.InvalidImageError):
                ps.preprocess(data)
        with patch.object(ps, "PRODUCT_MAX_PIXELS", 100 * 100 - 1), self.assertRaises(ps.InvalidImageError):
            ps.preprocess(encode(prenda(size=(100, 100), box=(20, 20, 80, 80))))


# ----------------------------------------------------------------------------- servicio
class ClassifyTests(unittest.TestCase):
    def test_sin_modelo_no_inventa_categoria(self):
        with patch.object(ps, "_get_model", return_value=None):
            r = ps.classify(encode(prenda()))
        self.assertFalse(r.modelo_entrenado)
        self.assertEqual((r.top3, r.confianza, r.baja_confianza), ([], 0.0, False))

    def test_top3_ordenado_y_coherente(self):
        probs = [0.02, 0.05, 0.6, 0.01, 0.2, 0.02, 0.05, 0.02, 0.02, 0.01]
        with patch.object(ps, "_get_model", return_value=modelo_falso(probs)):
            r = ps.classify(encode(prenda()))
        self.assertTrue(r.modelo_entrenado)
        self.assertEqual(r.categoria, ps.CLASES[2])
        self.assertEqual([s.categoria for s in r.top3], [ps.CLASES[2], ps.CLASES[4], ps.CLASES[1]])
        confs = [s.confianza for s in r.top3]
        self.assertEqual(confs, sorted(confs, reverse=True))
        self.assertAlmostEqual(r.confianza, 0.6, places=3)
        self.assertLessEqual(sum(confs), 1.0 + 1e-6)

    def test_aviso_de_baja_confianza_segun_umbral(self):
        def con_maximo(p):
            resto = (1 - p) / 9
            return [p] + [resto] * 9
        casos = [(0.95, False), (0.70, False), (0.69, True), (0.3, True)]
        for p, esperado in casos:
            with self.subTest(confianza=p), patch.object(ps, "_get_model", return_value=modelo_falso(con_maximo(p))):
                r = ps.classify(encode(prenda()))
                self.assertEqual(r.baja_confianza, esperado)
                self.assertEqual(r.umbral_confianza, ps.PRODUCT_CONFIDENCE_THRESHOLD)
        with patch.object(ps, "PRODUCT_CONFIDENCE_THRESHOLD", 0.99), \
                patch.object(ps, "_get_model", return_value=modelo_falso(con_maximo(0.95))):
            self.assertTrue(ps.classify(encode(prenda())).baja_confianza)

    def test_clases_iguales_en_entrenamiento_y_backend(self):
        """El modelo devuelve índices: el orden de CLASES debe coincidir con el del entrenamiento."""
        self.assertEqual(ps.CLASES, tp.CLASES)
        self.assertEqual((ps.MEAN, ps.STD), (tp.MEAN, tp.STD))
        self.assertEqual(len(ps.CLASES), 10)


# ----------------------------------------------------------------------------- API
class ApiTests(unittest.TestCase):
    def setUp(self):
        probs = [0.9] + [0.1 / 9] * 9
        patcher = patch.object(ps, "_get_model", return_value=modelo_falso(probs))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_imagen_valida_devuelve_contrato_completo(self):
        status, body = asyncio.run(post_file(encode(prenda(), "JPEG"), "image/jpeg", filename="foto.jpg"))
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {"categoria", "confianza", "top3", "baja_confianza", "umbral_confianza",
                                     "modelo_entrenado"})
        self.assertEqual(body["categoria"], ps.CLASES[0])
        self.assertEqual(set(body["top3"][0]), {"categoria", "confianza"})

    def test_errores_de_entrada(self):
        casos = [
            ("no es imagen", b"hola", "text/plain", 400, "debe ser una imagen"),
            ("vacío", b"", "image/png", 400, "vacío"),
            ("dañado", b"\x89PNG\r\n\x1a\n roto", "image/png", 400, "no es una imagen válida"),
            ("sin prenda", encode(Image.new("RGB", (64, 64), "white")), "image/png", 422, "ninguna prenda"),
        ]
        for nombre, data, ctype, esperado, mensaje in casos:
            with self.subTest(nombre):
                status, body = asyncio.run(post_file(data, ctype))
                self.assertEqual(status, esperado)
                self.assertIn(mensaje, body["detail"])

    def test_limite_de_tamaño(self):
        data = encode(prenda())
        with patch.object(route, "PRODUCT_MAX_BYTES", len(data) - 1):
            status, body = asyncio.run(post_file(data))
        self.assertEqual(status, 413)
        self.assertIn("supera el máximo", body["detail"])
        with patch.object(route, "PRODUCT_MAX_BYTES", len(data)):
            self.assertEqual(asyncio.run(post_file(data))[0], 200)

    def test_campo_obligatorio_imagen(self):
        status, body = asyncio.run(post_file(encode(prenda()), field="archivo"))
        self.assertEqual(status, 422)
        self.assertEqual(body["detail"][0]["loc"], ["body", "imagen"])


# ----------------------------------------------------------------------------- modelo real
@unittest.skipUnless(HAS_MODEL, "Se requiere backend/app/models/product_cnn.pt (python ml/train_products.py)")
class RealModelTests(unittest.TestCase):
    def test_fixtures_de_las_pruebas_e2e(self):
        botin = ps.classify((FIXTURES / "prenda-botin.png").read_bytes())
        self.assertEqual((botin.categoria, botin.baja_confianza), ("Botín", False))
        self.assertTrue(ps.classify((FIXTURES / "prenda-ambigua.png").read_bytes()).baja_confianza)

    def test_ejemplos_del_frontend_coinciden_con_su_nombre(self):
        """Los ejemplos son fotos reales a color (no Fashion-MNIST): prueban la generalización del pipeline."""
        nombres = {"camisa": "Camisa", "pantalon": "Pantalón", "vestido": "Vestido",
                   "zapatilla": "Zapatilla", "bolso": "Bolso", "botin": "Botín"}
        self.assertEqual({p.stem for p in EJEMPLOS.iterdir() if p.is_file()}, set(nombres))
        for slug, categoria in nombres.items():
            with self.subTest(slug):
                r = ps.classify((EJEMPLOS / f"{slug}.jpg").read_bytes())
                self.assertEqual(r.categoria, categoria)
                self.assertFalse(r.baja_confianza)

    def test_artefacto_coincide_con_metricas_publicadas(self):
        import hashlib
        metricas = json.loads((ROOT / "docs" / "clasificacion_productos_metricas.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(ps.PRODUCT_MODEL.read_bytes()).hexdigest(), metricas["artifact"]["sha256"])
        self.assertTrue(metricas["acceptance"]["passed"])


# ----------------------------------------------------------------------------- entrenamiento
class TrainingProtocolTests(unittest.TestCase):
    def test_validacion_estratificada_disjunta_y_reproducible(self):
        labels = np.repeat(np.arange(10), 100)
        train, val = tp.split_indices(labels, seed=42, val_size=200)
        self.assertEqual((len(train), len(val)), (800, 200))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(np.bincount(labels[val]).tolist(), [20] * 10)
        train2, val2 = tp.split_indices(labels, seed=42, val_size=200)
        np.testing.assert_array_equal(val, val2)
        self.assertFalse(np.array_equal(val, tp.split_indices(labels, seed=7, val_size=200)[1]))

    def test_submuestra_limita_validacion(self):
        labels = np.repeat(np.arange(10), 100)
        train, val = tp.split_indices(labels, seed=0, val_size=6000, subset=300)
        self.assertEqual((len(train), len(val)), (240, 60))

    def test_arquitectura_exportable_a_torchscript(self):
        model = tp.ProductCNN().eval()
        x = torch.randn(2, 1, 28, 28)
        with tempfile.TemporaryDirectory() as temp:
            sha = tp.export(model, Path(temp) / "m.pt", [(x, torch.zeros(2))])
            self.assertEqual(len(sha), 64)
        self.assertEqual(model(x).shape, (2, 10))

    @unittest.skipUnless(HAS_DATA, "Se requiere Fashion-MNIST en data/fashion_mnist (se descarga al entrenar)")
    def test_entrenamiento_rapido_genera_artefacto_y_metricas(self):
        with tempfile.TemporaryDirectory() as temp:
            out, met = Path(temp) / "m.pt", Path(temp) / "m.json"
            args = tp.parse_args(["--subset", "500", "--epochs", "1", "--output", str(out), "--metrics", str(met)])
            with patch("builtins.print"):
                summary = tp.main(args)
            self.assertTrue(out.is_file())
            guardado = json.loads(met.read_text(encoding="utf-8"))
        self.assertEqual(guardado["test"]["accuracy"], summary["test"]["accuracy"])
        self.assertEqual(guardado["protocol"]["split"]["subset"], 500)
        self.assertEqual(len(guardado["test"]["confusion_matrix"]["rows_true_cols_pred"]), 10)
        self.assertEqual(guardado["selected_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
