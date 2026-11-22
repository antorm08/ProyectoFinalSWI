"""Resumen extractivo de texto con TextRank (TF-IDF + PageRank sobre oraciones).

Funciona con español y portugués (las reseñas de Olist están en portugués).
Puede sustituirse por un modelo abstractivo (p. ej. mT5) manteniendo la misma interfaz.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.db import olist
from app.schemas import SummaryOutput

STOPWORDS = sorted(set("""
a ao aos as com da das de do dos e é em entre era essa esse esta este eu foi for isso já la las le lo los
mais mas me meu minha muito na nas não nem no nos o os ou para pela pelo por que se sem seu sua são também
te tem ter um uma umas uns você del el en es está estaba ha hay la más mi muy ni pero porque su sus tu un una y ya
""".split()))

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Divide en oraciones, descarta las muy cortas y elimina duplicadas (frecuentes en reseñas)."""
    seen, out = set(), []
    for s in (s.strip() for s in _SPLIT.split(text)):
        key = s.lower().rstrip(".!? ")
        if len(s.split()) >= 3 and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def textrank(sentences: list[str], n: int, damping: float = 0.85, iters: int = 50) -> list[str]:
    if len(sentences) <= n:
        return sentences
    tfidf = TfidfVectorizer(stop_words=STOPWORDS, lowercase=True).fit_transform(sentences)
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0)
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    m = sim / row_sums
    scores = np.ones(len(sentences)) / len(sentences)
    for _ in range(iters):
        scores = (1 - damping) / len(sentences) + damping * m.T @ scores
    top = sorted(np.argsort(scores)[::-1][:n])          # conservar el orden original
    return [sentences[i] for i in top]


def summarize(text: str, n: int = 3) -> SummaryOutput:
    sentences = split_sentences(text)
    chosen = textrank(sentences, n)
    return SummaryOutput(resumen=" ".join(chosen), oraciones_originales=len(sentences),
                         oraciones_resumen=len(chosen))


def summarize_seller(seller_id: str, n: int = 5) -> SummaryOutput | None:
    """Junta todas las reseñas de un vendedor y las resume."""
    if not (olist.available("reviews") and olist.available("items")):
        return None
    items = olist.load("items")
    orders = items.loc[items["seller_id"] == seller_id, "order_id"].unique()
    reviews = olist.load("reviews")
    reviews = reviews[reviews["order_id"].isin(orders)]
    textos = reviews["review_comment_message"].dropna().astype(str)
    if textos.empty:
        return None
    texto = "\n".join(t if t.rstrip().endswith((".", "!", "?")) else t + "." for t in textos)
    return summarize(texto, n)
