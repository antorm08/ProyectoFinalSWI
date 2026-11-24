"""Componente 5: limpieza, temas, resumen extractivo y API de reseñas (datos sintéticos y, si existen, reales)."""
import asyncio
import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.main import app
from app.services import summary_service as ss

OLIST = ROOT / "data" / "olist"


async def call(method, path, body=None, query=""):
    """Invoca la app FastAPI real por ASGI, sin servidor ni dependencias extra."""
    events = []
    payload = json.dumps(body).encode() if body is not None else b""

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(event):
        events.append(event)

    await app({
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
        "root_path": "", "query_string": query.encode(), "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234), "server": ("test", 80),
    }, receive, send)
    status = next(e["status"] for e in events if e["type"] == "http.response.start")
    return status, json.loads(b"".join(e.get("body", b"") for e in events if e["type"] == "http.response.body"))


def synthetic_olist():
    """Vendedor s1: muchos reclamos de entrega. s2: casi todo elogios. Sin texto en algunas reseñas."""
    rows = [
        ("s1", 1, "Não recebi o produto."), ("s1", 1, "nao recebi o produto!"), ("s1", 1, "Ainda não recebi o produto"),
        ("s1", 2, "O pedido atrasou muito e ninguém responde o e-mail."), ("s1", 1, "Veio quebrado, quero devolver."),
        ("s1", 2, "Produto diferente da foto do anúncio."), ("s1", 5, "Chegou antes do prazo, recomendo."),
        ("s1", 4, None), ("s1", 3, "Produto ok, nada especial."),
        ("s2", 5, "Entrega rápida e produto de ótima qualidade."), ("s2", 5, "Chegou antes do prazo! Recomendo a loja."),
        ("s2", 4, "Produto muito bom e bem embalado."), ("s2", 5, None), ("s2", 1, "Faltou um item do kit."),
    ]
    reviews = pd.DataFrame({
        "review_id": [f"r{i}" for i in range(len(rows))], "order_id": [f"o{i}" for i in range(len(rows))],
        "review_score": [r[1] for r in rows], "review_comment_title": None,
        "review_comment_message": [r[2] for r in rows],
    })
    items = pd.DataFrame({"order_id": reviews["order_id"], "order_item_id": 1, "seller_id": [r[0] for r in rows],
                          "product_id": ["p1" if r[0] == "s1" else "p2" for r in rows]})
    products = pd.DataFrame({"product_id": ["p1", "p2"], "product_category_name": ["moveis_decoracao", "beleza_saude"]})
    return reviews, items, products


class TextTests(unittest.TestCase):
    def test_clean_text_fixes_glued_punctuation_and_line_breaks(self):
        self.assertEqual(ss.clean_text("Vieram iguais.Paguei   caro\r\nNão gostei"), "Vieram iguais. Paguei caro\nNão gostei")
        self.assertEqual(ss.clean_text("Nota 4.5 de 5"), "Nota 4.5 de 5")      # decimales intactos

    def test_split_sentences_drops_short_and_normalized_duplicates(self):
        text = "Muito bom. Não recebi o produto. nao recebi o produto!! O vendedor não responde."
        self.assertEqual(ss.split_sentences(text), ["Não recebi o produto.", "O vendedor não responde."])
        self.assertEqual(ss.split_sentences("ok. top."), [])

    def test_themes_portuguese_spanish_and_ambiguous_words(self):
        cases = {
            "O pedido atrasou e chegou depois do prazo": "plazo_entrega",
            "Ainda não recebi meu pedido": "no_recibido",
            "No recibí el producto": "no_recibido",
            "Comprei dois e só veio um": "incompleto",
            "Produto diferente da foto": "anuncio",
            "Veio com defeito, não funciona": "calidad",
            "La calidad es mala": "calidad",
            "O SAC não responde meus e-mails": "atencion",
            "Não me deram nenhuma satisfação": "atencion",
            "Quero devolver e ter o reembolso": "devolucion",
            "A caixa chegou amassada": "embalaje",
            "Ótimo custo benefício": "precio",
            "Recomendo, fiquei satisfeita": "satisfaccion",
        }
        for text, theme in cases.items():
            with self.subTest(text=text):
                self.assertIn(theme, ss.detect_themes(text))
        self.assertNotIn("calidad", ss.detect_themes("Loja péssima"))           # opinión de la tienda, no del producto
        self.assertNotIn("atencion", ss.detect_themes("O suporte da cortina é firme"))
        self.assertNotIn("precio", ss.detect_themes("Paguei e não recebi"))
        self.assertEqual(ss.theme_matrix(pd.Series([None, ""], dtype=object)).sum().sum(), 0)

    def test_sentence_pool_counts_each_review_once(self):
        pool = ss.sentence_pool(["Não recebi o produto. Não recebi o produto.", "nao recebi o produto", "Entrega muito rápida mesmo"])
        counts = dict(zip(pool["oracion"], pool["repeticiones"]))
        self.assertEqual(counts, {"Não recebi o produto.": 2, "Entrega muito rápida mesmo": 1})


class RankingTests(unittest.TestCase):
    def test_textrank_is_a_distribution_and_weights_shift_ranking(self):
        sim = np.array([[0, .5, .1, 0], [.5, 0, .1, 0], [.1, .1, 0, 0], [0, 0, 0, 0]], dtype=float)
        plain = ss.textrank_scores(sim)
        self.assertAlmostEqual(plain.sum(), 1.0, places=6)
        self.assertTrue(np.all(plain > 0))                                        # nodo aislado no rompe nada
        weighted = ss.textrank_scores(sim, np.array([1, 1, 1, 50.0]))
        self.assertGreater(weighted[3], plain[3])

    def test_mmr_prefers_new_ideas_over_paraphrases(self):
        pool = pd.DataFrame({"oracion": [
            "Não recebi o produto", "Não recebi o produto ainda", "Ainda não recebi o produto", "Eu não recebi o produto",
            "Produto chegou quebrado e com defeito", "O atendimento não responde os e-mails", "Não recebi meu produto"],
            "repeticiones": [9, 8, 7, 6, 3, 3, 5]})
        themes = lambda idx: set().union(*(ss.detect_themes(pool.at[i, "oracion"]) for i in idx))
        plain = ss.rank_pool(pool, 3, method="textrank_ponderado")
        mmr = ss.rank_pool(pool, 3)                                                # configuración de producción
        self.assertEqual(themes(plain), {"no_recibido"})                         # tres versiones de lo mismo
        self.assertEqual(themes(mmr), {"no_recibido", "calidad", "atencion"})
        self.assertEqual(mmr[0], 0)                                              # la frase más repetida, primero

    def test_paraphrase_threshold_only_applies_while_alternatives_remain(self):
        sim = np.array([[1, .9, .9], [.9, 1, .9], [.9, .9, 1]])
        self.assertEqual(sorted(ss.mmr_select(np.array([3, 2, 1.0]), sim, 3)), [0, 1, 2])

    def test_rank_pool_small_and_empty(self):
        self.assertEqual(ss.rank_pool(pd.DataFrame({"oracion": [], "repeticiones": []}), 3), [])
        pool = pd.DataFrame({"oracion": ["uma frase com conteúdo", "outra frase diferente aqui"], "repeticiones": [1, 4]})
        self.assertEqual(ss.rank_pool(pool, 5), [1, 0])

    def test_summarize_text_keeps_order_and_limits(self):
        text = ("El producto tiene buena calidad. La entrega fue puntual y segura. "
                "El embalaje protege bien el producto. Recomiendo comprar este producto.")
        out = ss.summarize(text, 2)
        self.assertEqual((out.oraciones_originales, out.oraciones_resumen), (4, 2))
        positions = [text.index(s) for s in ss.split_sentences(out.resumen)]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("embalaje", {t.tema for t in out.temas})
        self.assertEqual(ss.summarize("ok ok. sí. no.", 3).resumen, "")


class GroupTests(unittest.TestCase):
    def setUp(self):
        self.table, self.links = ss.build_tables(*synthetic_olist())
        self.profile = ss.platform_profile(self.table)

    def test_build_tables_links_categories_and_themes(self):
        self.assertEqual(len(self.table), 14)
        self.assertEqual(int(self.table["texto"].notna().sum()), 12)
        self.assertEqual(set(self.links["categoria"]), {"moveis_decoracao", "beleza_saude"})
        self.assertTrue(set(ss.TEMAS) <= set(self.table.columns))

    def test_seller_insights_split_by_score_and_compare_with_platform(self):
        orders = self.links.loc[self.links["seller_id"] == "s1", "order_id"]
        ins = ss.analyze_group(self.table[self.table["order_id"].isin(orders)], 3, self.profile, "vendedor", "s1", "s1")
        self.assertEqual((ins.resenas_totales, ins.resenas_con_comentario), (9, 8))
        self.assertEqual(ins.distribucion_notas, {"1": 4, "2": 2, "3": 1, "4": 1, "5": 1})
        self.assertAlmostEqual(ins.pct_reclamos, 6 / 9, places=3)
        self.assertEqual(ins.reclamos[0].texto.lower().rstrip(".!"), "não recebi o produto")
        self.assertEqual(ins.reclamos[0].repeticiones, 2)
        reclamo_texts = " ".join(p.texto for p in ins.reclamos)
        self.assertNotIn("antes do prazo", reclamo_texts)                         # los elogios no se cuelan
        self.assertEqual(ins.temas_reclamos[0].tema, "no_recibido")
        self.assertIsNotNone(ins.temas_reclamos[0].porcentaje_plataforma)
        self.assertIn("por encima de la media", ins.resumen_ejecutivo)
        self.assertIn("pedido no recibido", ins.resumen_ejecutivo)
        self.assertIn("Hay pocas reseñas con comentario: el resumen es orientativo.", ins.advertencias)
        self.assertTrue(ins.resumen.startswith(ins.resumen_ejecutivo))

    def test_group_without_comments_returns_none(self):
        empty = self.table[self.table["texto"].isna()]
        self.assertIsNone(ss.analyze_group(empty, 3, self.profile, "vendedor", "x", "x"))


class ApiTests(unittest.TestCase):
    def setUp(self):
        table, links = ss.build_tables(*synthetic_olist())
        tables = (table, links, ss.platform_profile(table))
        self.patches = [patch.object(ss, "_tables", lambda: tables), patch.object(ss, "data_available", lambda: True)]
        for p in self.patches:
            p.start()
        ss.summarize_platform.cache_clear()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        ss.summarize_platform.cache_clear()

    def test_text_endpoint_is_backward_compatible(self):
        body = {"texto": "El producto tiene buena calidad. La entrega fue puntual. Recomiendo comprar este producto.",
                "n_oraciones": 2}
        status, data = asyncio.run(call("POST", "/api/reviews/summarize", body))
        self.assertEqual(status, 200)
        self.assertTrue({"resumen", "oraciones_originales", "oraciones_resumen", "temas"} <= set(data))
        self.assertEqual(data["oraciones_resumen"], 2)
        status, _ = asyncio.run(call("POST", "/api/reviews/summarize", {"texto": "corto", "n_oraciones": 2}))
        self.assertEqual(status, 422)

    def test_seller_category_overview_and_options(self):
        status, data = asyncio.run(call("GET", "/api/reviews/seller/s1", query="n_oraciones=2"))
        self.assertEqual(status, 200)
        self.assertEqual((data["ambito"], len(data["reclamos"])), ("vendedor", 2))
        status, data = asyncio.run(call("GET", "/api/reviews/category/beleza_saude"))
        self.assertEqual((status, data["etiqueta"], data["resenas_totales"]), (200, "Beleza saude", 5))
        status, data = asyncio.run(call("GET", "/api/reviews/overview", query="n_oraciones=3"))
        self.assertEqual((status, data["ambito"], data["resenas_totales"]), (200, "plataforma", 14))
        status, data = asyncio.run(call("GET", "/api/reviews/options"))
        self.assertEqual(status, 200)
        self.assertEqual([v["seller_id"] for v in data["vendedores"]], ["s1", "s2"])
        self.assertEqual(data["categorias"][0]["categoria"], "moveis_decoracao")

    def test_unknown_ids_and_invalid_lengths(self):
        self.assertEqual(asyncio.run(call("GET", "/api/reviews/seller/no-existe"))[0], 404)
        self.assertEqual(asyncio.run(call("GET", "/api/reviews/category/no-existe"))[0], 404)
        self.assertEqual(asyncio.run(call("GET", "/api/reviews/seller/s1", query="n_oraciones=16"))[0], 422)
        self.assertEqual(asyncio.run(call("GET", "/api/reviews/overview", query="n_oraciones=0"))[0], 422)

    def test_without_olist_data(self):
        with patch.object(ss, "data_available", lambda: False):
            self.assertEqual(asyncio.run(call("GET", "/api/reviews/seller/s1"))[0], 404)
            self.assertEqual(asyncio.run(call("GET", "/api/reviews/overview"))[0], 404)
            self.assertEqual(asyncio.run(call("GET", "/api/reviews/options"))[1], {"vendedores": [], "categorias": []})


@unittest.skipUnless(all((OLIST / f"olist_{n}_dataset.csv").exists() for n in ("order_reviews", "order_items")),
                     "Requiere los CSV de Olist")
class RealOlistReviewTests(unittest.TestCase):
    def test_platform_and_top_seller(self):
        ss.summarize_platform.cache_clear()
        platform = ss.summarize_platform(5)
        self.assertGreater(platform.resenas_con_comentario, 40_000)
        self.assertEqual((len(platform.reclamos), len(platform.elogios)), (5, 5))
        self.assertEqual(platform.temas_reclamos[0].tema, "no_recibido")
        top = ss.review_options(1).vendedores[0]
        start = time.perf_counter()
        seller = ss.summarize_seller(top.seller_id, 5)
        self.assertLess(time.perf_counter() - start, 3)
        self.assertEqual(seller.resenas_con_comentario, top.resenas_con_comentario)
        self.assertTrue(all(p.texto for p in seller.reclamos + seller.elogios))
