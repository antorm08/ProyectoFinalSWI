"""Resumen de opiniones y reclamos de clientes (componente 5).

Tres piezas, ninguna requiere entrenamiento y todas son explicables:

1. Limpieza y segmentación de las reseñas en oraciones (Olist está en portugués; también acepta español).
2. Detección de temas por léxico (entrega, calidad, atención…), insensible a tildes y mayúsculas.
3. Resumen extractivo: TextRank ponderado por frecuencia + MMR (Maximal Marginal Relevance) para no repetir
   la misma idea. Las reseñas de Olist se separan por nota en reclamos (1–2) y elogios (4–5).

Parámetros y comparación con alternativas: ml/evaluate_reviews.py → docs/evaluacion_resumen.md.
"""
import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app.db import olist
from app.schemas import (FraseRepresentativa, OpcionCategoria, OpcionVendedor, ReviewInsights, ReviewOptions,
                         SummaryOutput, TemaDetectado)

# ---- Parámetros (elegidos en docs/evaluacion_resumen.md) ----
MIN_PALABRAS = 3          # oraciones más cortas ("Muito bom") no aportan contenido al resumen
MMR_LAMBDA = 0.7          # 1 = solo relevancia, 0 = solo diversidad
MAX_SIMILITUD = 0.6       # coseno a partir del cual una frase se considera paráfrasis de otra ya elegida
MAX_CANDIDATAS = 400      # tope de oraciones que entran al grafo de TextRank (coste cuadrático)
DAMPING = 0.85
NOTA_RECLAMO = 2          # nota <= 2 → reclamo
NOTA_ELOGIO = 4           # nota >= 4 → elogio
POCAS_RESENAS = 10        # por debajo de esto el resumen se marca como orientativo

# ---- Temas: expresiones sobre texto en minúsculas y sin tildes (portugués y español) ----
TEMAS: dict[str, tuple[str, str]] = {
    "plazo_entrega": ("Plazo de entrega",
        r"\batras|\bdemor|\bprazo|\bprevis|\bretras|\btard|\brapid|\bagil|\bantes do|\ba tempo\b"
        r"|\bno tempo\b|\bplazo|\bentreg\w* (?:rapid|lent|agil)"),
    "no_recibido": ("Pedido no recibido",
        r"\b(?:nao|nunca|n) (?:recebi|recebemos|recebido|chegou|chegaram|foi entregue|foram entregues"
        r"|entregaram|entregou)\b|\bainda nao (?:recebi|chegou)|\bextravi|\bsem receber\b"
        r"|\bno (?:he )?recib|\bno (?:me )?(?:llego|ha llegado)|\bnunca llego|\baguardando (?:o produto"
        r"|a entrega)"),
    "incompleto": ("Pedido incompleto",
        r"\bfalt(?:ou|ando|a|aram|am|o)\b|\bincomplet|\bnao (?:veio|vieram)\b|\b(?:so|apenas|somente"
        r"|solo) (?:veio|vieram|recebi|chegou|chegaram|llego)\b|\bquantidade\b|\bmetade\b|\bcantidad\b"),
    "anuncio": ("Fidelidad al anuncio",
        r"\bdiferente|\berrad|\boutro produto|\boutra cor|\bcondiz|\bconforme\b|\bcomo (?:anunciad|descrit"
        r"|esperad|mostrad)|\banunci|\bdescri(?:cao|to|ta|pcion)\b|\bfoto|\bimagem|\bequivocad"
        r"|\bcorrespond|\bespecifica|\btamanho\b|\btalla\b"),
    "calidad": ("Calidad y estado del producto",
        r"\bqualidad|\bcalidad|\bdefeit|\bdefect|\bquebr|\bdanific|\bestragad|\brasgad|\bamassad|\briscad"
        r"|\bfunciona|\bfuncionou|\bmaterial|\bfragil|\bresistent|\bdurabil|\bfalsific|\boriginal\b"
        r"|\bpirata|\bvagabund|\bproduto (?:\w+ )?(?:ruim|pessimo)|\bvencid|\broto\b|\brota\b|\bacabamento"
        r"|\bbem feito|\bmal feito"),
    "atencion": ("Atención al cliente",
        r"\batendiment|\batencion|\bcontat|\bcontact|\bresp(?:ond|ost)|\bretorno\b|\bsac\b|\bsoporte"
        r"|\be-?mails?\b|\bcorreo\b|\bwhats|\btelefon|\bligu|\bligac|\breclam|\bsatisfa(?:cao|coes)"
        r"|\bsolucion|\bresolv|\bprotocolo|\bchat\b|\bmensag|\bmensaj|\bcomunica"),
    "devolucion": ("Cambios y devoluciones",
        r"\bdevol|\btroc(?:a|ar|ou|ado|ada)\b|\breembols|\bestorn|\bcancel|\bdinheiro de volta|\bressarc"
        r"|\bcambiar|\breemplaz|\bpostagem"),
    "embalaje": ("Embalaje",
        r"\bembala|\bempaque|\bempacot|\bcaixa|\bcaja\b|\bpacote|\bpaquete|\blacrad|\bplastico bolha"
        r"|\bprotegid"),
    "precio": ("Precio",
        r"\bpreco|\bprecio|\bbarat|\bcaro\b|\bcusto|\bcosto|\bvalor\b|\bbeneficio|\bpromoc|\bfrete"
        r"|\bdesconto|\bdescuento|\bcobra"),
    "satisfaccion": ("Satisfacción y recomendación",
        r"\brecomend|\bsatisfeit|\bsatisfech|\bsuperou|\badorei|\bamei\b|\bgostei|\bencant|\bexcelent"
        r"|\botim[oa]|\bperfeit|\bparabens|\bfelicit|\bmaravilh|\bconfiav|\bconfiable|\bvoltarei"
        r"|\bcomprarei|\bvolvere"),
}
# "Satisfacción" es genérico: se muestra en los gráficos pero no se destaca en el resumen ejecutivo.
TEMAS_GENERICOS = {"satisfaccion"}
_TEMAS_RE = {key: re.compile(pattern) for key, (_, pattern) in TEMAS.items()}

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_PUNT_PEGADA = re.compile(r"([.!?;,])(?=[^\W\d_])")    # "iguais.Paguei" → "iguais. Paguei"
_ESPACIOS = re.compile(r"[ \t ]+")
_NO_PALABRA = re.compile(r"[^\w\s]")


# ---- Texto ----
def normalize(text: str) -> str:
    """Minúsculas, sin tildes y con espacios simples (para comparar y detectar temas)."""
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return _ESPACIOS.sub(" ", text).strip()


def clean_text(text: str) -> str:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = _PUNT_PEGADA.sub(r"\1 ", text)
    return "\n".join(_ESPACIOS.sub(" ", line).strip() for line in text.split("\n")).strip()


def sentence_key(sentence: str) -> str:
    """Clave para reconocer la misma frase escrita con otra puntuación, tildes o mayúsculas."""
    return " ".join(_NO_PALABRA.sub(" ", normalize(sentence)).split())


def split_sentences(text: str) -> list[str]:
    """Divide en oraciones, descarta las muy cortas y elimina duplicadas (frecuentes en reseñas)."""
    seen, out = set(), []
    for s in (s.strip() for s in _SPLIT.split(clean_text(text))):
        key = sentence_key(s)
        if len(key.split()) >= MIN_PALABRAS and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def detect_themes(text: str) -> list[str]:
    norm = normalize(text)
    return [key for key, regex in _TEMAS_RE.items() if regex.search(norm)]


def theme_matrix(texts: pd.Series) -> pd.DataFrame:
    """Una columna booleana por tema; los textos vacíos no tienen ningún tema."""
    norm = texts.map(lambda t: normalize(t) if isinstance(t, str) else "")
    return pd.DataFrame({key: norm.str.contains(regex) for key, regex in _TEMAS_RE.items()}, index=texts.index)


def theme_shares(themes: pd.DataFrame, baseline: dict[str, float] | None = None) -> list[TemaDetectado]:
    """Porcentaje de textos que mencionan cada tema, de mayor a menor (solo temas presentes)."""
    if themes.empty:
        return []
    counts = themes.sum()
    out = [TemaDetectado(tema=key, etiqueta=TEMAS[key][0], menciones=int(counts[key]),
                         porcentaje=round(float(counts[key] / len(themes)), 4),
                         porcentaje_plataforma=None if baseline is None else round(baseline.get(key, 0.0), 4))
           for key in TEMAS if counts[key] > 0]
    return sorted(out, key=lambda t: -t.menciones)


# ---- Resumen extractivo ----
def sentence_pool(texts: Iterable[str]) -> pd.DataFrame:
    """Oraciones únicas de un conjunto de reseñas y en cuántas reseñas aparece cada una."""
    first: dict[str, str] = {}
    counts: dict[str, int] = {}
    for text in texts:
        for s in split_sentences(text):                  # ya sin duplicadas dentro de la reseña
            key = sentence_key(s)
            first.setdefault(key, s)
            counts[key] = counts.get(key, 0) + 1
    return pd.DataFrame({"oracion": list(first.values()), "repeticiones": [counts[k] for k in first]})


def vectorize(sentences: list[str]):
    """TF-IDF de n-gramas de caracteres: tolera faltas de ortografía, tildes y variaciones de palabras."""
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                           preprocessor=normalize).fit_transform(sentences)


def textrank_scores(sim: np.ndarray, weights: np.ndarray | None = None, damping: float = DAMPING,
                    tol: float = 1e-8, max_iter: int = 200) -> np.ndarray:
    """PageRank sobre el grafo de similitud. Con `weights`, el salto aleatorio favorece las frases frecuentes."""
    n = sim.shape[0]
    sim = sim.copy()
    np.fill_diagonal(sim, 0)
    rows = sim.sum(axis=1)
    dangling = rows == 0
    transition = sim / np.where(dangling, 1, rows)[:, None]
    p = np.full(n, 1 / n) if weights is None else weights / weights.sum()
    scores = p.copy()
    for _ in range(max_iter):
        new = (1 - damping) * p + damping * (transition.T @ scores + scores[dangling].sum() * p)
        if np.abs(new - scores).sum() < tol:
            return new
        scores = new
    return scores


def mmr_select(relevance: np.ndarray, sim: np.ndarray, n: int, lam: float = MMR_LAMBDA,
               max_sim: float | None = None) -> list[int]:
    """Maximal Marginal Relevance: elige frases relevantes y distintas de las ya elegidas.

    Con `max_sim` descarta además las paráfrasis (similitud > max_sim con una frase ya elegida) mientras
    queden alternativas.
    """
    max_sim = MAX_SIMILITUD if max_sim is None else max_sim
    relevance = relevance / relevance.max() if relevance.max() > 0 else relevance
    selected = [int(np.argmax(relevance))]
    while len(selected) < min(n, len(relevance)):
        closest = sim[:, selected].max(axis=1)
        score = lam * relevance - (1 - lam) * closest
        score[selected] = -np.inf
        if np.isfinite(score[closest <= max_sim]).any():
            score[closest > max_sim] = -np.inf
        selected.append(int(np.argmax(score)))
    return selected


def rank_pool(pool: pd.DataFrame, n: int, method: str = "textrank_mmr", lam: float = MMR_LAMBDA,
              max_candidates: int = MAX_CANDIDATAS, max_sim: float | None = None) -> list[int]:
    """Índices de `pool` elegidos para el resumen, del más al menos relevante.

    Métodos (comparados en la evaluación): centroide, textrank, textrank_ponderado, textrank_mmr.
    """
    if pool.empty:
        return []
    if len(pool) <= n:
        return [int(i) for i in np.argsort(-pool["repeticiones"].to_numpy(), kind="stable")]
    X = vectorize(pool["oracion"].tolist())
    weights = pool["repeticiones"].to_numpy(dtype=float)
    centroid = np.asarray(X.T @ weights).ravel() / weights.sum()
    centrality = X @ centroid
    if method == "centroide":
        return [int(i) for i in np.argsort(-centrality, kind="stable")[:n]]
    candidates = np.argsort(-(centrality * np.log1p(weights)), kind="stable")[:max_candidates]
    Xc = X[candidates]
    sim = (Xc @ Xc.T).toarray()
    scores = textrank_scores(sim, None if method == "textrank" else weights[candidates])
    if method == "textrank_mmr":
        chosen = mmr_select(scores, sim, n, lam, max_sim)
    else:
        chosen = list(np.argsort(-scores, kind="stable")[:n])
    return [int(candidates[i]) for i in chosen]


def summarize(text: str, n: int = 3) -> SummaryOutput:
    """Resumen de un texto pegado por el usuario: conserva el orden original de las oraciones."""
    sentences = split_sentences(text)
    pool = pd.DataFrame({"oracion": sentences, "repeticiones": [1] * len(sentences)})
    chosen = sorted(rank_pool(pool, n))
    themes = theme_matrix(pd.Series(sentences, dtype=object))
    return SummaryOutput(resumen=" ".join(sentences[i] for i in chosen), oraciones_originales=len(sentences),
                         oraciones_resumen=len(chosen), temas=theme_shares(themes))


# ---- Reseñas de Olist ----
def category_label(categoria: str) -> str:
    return categoria.replace("_", " ").strip().capitalize()


def build_tables(reviews: pd.DataFrame, items: pd.DataFrame,
                 products: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepara (reseñas, vínculos pedido→vendedor/categoría).

    reseñas: order_id, nota, texto (limpio o None) y una columna booleana por tema.
    """
    r = reviews[["review_id", "order_id", "review_score", "review_comment_message"]].drop_duplicates(
        ["review_id", "order_id"])
    texto = r["review_comment_message"].map(lambda t: (clean_text(t) or None) if isinstance(t, str) else None)
    table = pd.DataFrame({"order_id": r["order_id"].to_numpy(), "nota": r["review_score"].astype(int).to_numpy(),
                          "texto": texto.to_numpy(dtype=object)})
    table = pd.concat([table, theme_matrix(table["texto"])], axis=1)
    links = items[["order_id", "seller_id"] + (["product_id"] if "product_id" in items else [])]
    if products is not None and "product_id" in links:
        links = links.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
        links = links.rename(columns={"product_category_name": "categoria"})
    else:
        links = links.assign(categoria=np.nan)
    return table, links[["order_id", "seller_id", "categoria"]].drop_duplicates()


def platform_profile(table: pd.DataFrame) -> dict:
    """Indicadores de toda la plataforma, para comparar a un vendedor o a una categoría."""
    commented = table["texto"].notna()
    reclamos = table[commented & (table["nota"] <= NOTA_RECLAMO)]
    return {"pct_reclamos": float((table["nota"] <= NOTA_RECLAMO).mean()),
            "temas_reclamos": {k: float(reclamos[k].mean()) if len(reclamos) else 0.0 for k in TEMAS}}


def _phrases(texts: pd.Series, n: int) -> tuple[list[FraseRepresentativa], int]:
    pool = sentence_pool(texts)
    chosen = rank_pool(pool, n)
    return [FraseRepresentativa(texto=pool.at[i, "oracion"], repeticiones=int(pool.at[i, "repeticiones"]),
                                temas=detect_themes(pool.at[i, "oracion"])) for i in chosen], len(pool)


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _num(x: int) -> str:
    """Mismo formato que el frontend (Intl "es"): 99.224."""
    return f"{x:,}".replace(",", ".")


def _dec(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def _list_es(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " y " + parts[-1]


def executive_summary(ins: dict, profile: dict) -> str:
    """Resumen en español generado a partir de los indicadores (plantilla, sin inventar contenido)."""
    out = [f"Se analizaron {_num(ins['resenas_totales'])} reseñas ({_num(ins['resenas_con_comentario'])} con "
           f"comentario), con una nota media de {_dec(ins['nota_media'])} sobre 5.",
           f"El {_pct(ins['pct_reclamos'])} son reclamos (nota 1–2) y el {_pct(ins['pct_elogios'])} son elogios "
           f"(nota 4–5)."]
    if ins["ambito"] != "plataforma":
        diff = ins["pct_reclamos"] - profile["pct_reclamos"]
        nivel = "es similar a" if abs(diff) < 0.03 else ("está por encima de" if diff > 0 else "está por debajo de")
        out.append(f"La tasa de reclamos {nivel} la media de la plataforma ({_pct(profile['pct_reclamos'])}).")
    top_rec = [t for t in ins["temas_reclamos"] if t.tema not in TEMAS_GENERICOS and t.menciones >= 2
               and t.porcentaje >= 0.05][:3]
    if top_rec:
        out.append("Los reclamos se concentran en: "
                   + _list_es([f"{t.etiqueta.lower()} ({_pct(t.porcentaje)})" for t in top_rec]) + ".")
    elif ins["n_reclamos_comentados"]:
        out.append("Los reclamos no se concentran en un tema concreto.")
    if ins["ambito"] != "plataforma":
        excess = [t for t in ins["temas_reclamos"] if t.tema not in TEMAS_GENERICOS and t.menciones >= 3
                  and t.porcentaje_plataforma and t.porcentaje >= 1.5 * t.porcentaje_plataforma
                  and t.porcentaje - t.porcentaje_plataforma >= 0.05]
        if excess:
            t = max(excess, key=lambda t: t.porcentaje - t.porcentaje_plataforma)
            out.append(f"Destaca frente a la plataforma: {t.etiqueta.lower()} aparece en el {_pct(t.porcentaje)} "
                       f"de sus reclamos (media: {_pct(t.porcentaje_plataforma)}).")
    top_elo = [t for t in ins["temas_elogios"] if t.tema not in TEMAS_GENERICOS and t.menciones >= 2
               and t.porcentaje >= 0.05][:2]
    if top_elo:
        out.append("Lo más valorado: " + _list_es([f"{t.etiqueta.lower()} ({_pct(t.porcentaje)})" for t in top_elo])
                   + ".")
    return " ".join(out)


def analyze_group(group: pd.DataFrame, n: int, profile: dict, ambito: str, identificador: str | None,
                  etiqueta: str) -> ReviewInsights | None:
    """Indicadores, temas y frases representativas de un conjunto de reseñas. None si no hay comentarios."""
    commented = group["texto"].notna()
    if not commented.any():
        return None
    notas = group["nota"]
    rec = group[commented & (notas <= NOTA_RECLAMO)]
    elo = group[commented & (notas >= NOTA_ELOGIO)]
    keys = list(TEMAS)
    reclamos, pool_rec = _phrases(rec["texto"], n)
    elogios, pool_elo = _phrases(elo["texto"], n)
    ins = {
        "ambito": ambito, "identificador": identificador, "etiqueta": etiqueta,
        "resenas_totales": int(len(group)), "resenas_con_comentario": int(commented.sum()),
        "nota_media": round(float(notas.mean()), 2),
        "distribucion_notas": {str(k): int((notas == k).sum()) for k in range(1, 6)},
        "pct_reclamos": round(float((notas <= NOTA_RECLAMO).mean()), 4),
        "pct_elogios": round(float((notas >= NOTA_ELOGIO).mean()), 4),
        "pct_reclamos_plataforma": round(profile["pct_reclamos"], 4),
        "temas": theme_shares(group.loc[commented, keys]),
        "temas_reclamos": theme_shares(rec[keys], profile["temas_reclamos"]),
        "temas_elogios": theme_shares(elo[keys]),
        "reclamos": reclamos, "elogios": elogios, "n_reclamos_comentados": len(rec),
    }
    advertencias = []
    if ins["resenas_con_comentario"] < POCAS_RESENAS:
        advertencias.append("Hay pocas reseñas con comentario: el resumen es orientativo.")
    if not reclamos:
        advertencias.append("No hay reclamos con comentario en este conjunto.")
    if not elogios:
        advertencias.append("No hay elogios con comentario en este conjunto.")
    ejecutivo = executive_summary(ins, profile)
    lines = [ejecutivo]
    for title, phrases in (("Reclamos frecuentes", reclamos), ("Lo más valorado", elogios)):
        if phrases:
            lines += ["", f"{title}:"] + [f"- {p.texto}" + (f" (×{p.repeticiones})" if p.repeticiones > 1 else "")
                                          for p in phrases]
    ins.pop("n_reclamos_comentados")
    return ReviewInsights(**ins, resumen_ejecutivo=ejecutivo, resumen="\n".join(lines), advertencias=advertencias,
                          oraciones_originales=pool_rec + pool_elo, oraciones_resumen=len(reclamos) + len(elogios))


@lru_cache(maxsize=1)
def _tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    products = olist.load("products") if olist.available("products") else None
    table, links = build_tables(olist.load("reviews"), olist.load("items"), products)
    return table, links, platform_profile(table)


def data_available() -> bool:
    return olist.available("reviews") and olist.available("items")


def summarize_seller(seller_id: str, n: int = 5) -> ReviewInsights | None:
    """Junta todas las reseñas de un vendedor y resume reclamos y elogios."""
    if not data_available():
        return None
    table, links, profile = _tables()
    orders = links.loc[links["seller_id"] == seller_id, "order_id"].unique()
    group = table[table["order_id"].isin(orders)]
    return analyze_group(group, n, profile, "vendedor", seller_id, f"Vendedor {seller_id[:8]}…")


def summarize_category(categoria: str, n: int = 5) -> ReviewInsights | None:
    if not data_available():
        return None
    table, links, profile = _tables()
    orders = links.loc[links["categoria"] == categoria, "order_id"].unique()
    group = table[table["order_id"].isin(orders)]
    return analyze_group(group, n, profile, "categoria", categoria, category_label(categoria))


@lru_cache(maxsize=16)
def summarize_platform(n: int = 5) -> ReviewInsights | None:
    if not data_available():
        return None
    table, _, profile = _tables()
    return analyze_group(table, n, profile, "plataforma", None, "Toda la plataforma")


def review_options(limit: int = 30) -> ReviewOptions:
    """Vendedores y categorías con más reseñas comentadas, para elegir sin conocer el identificador."""
    if not data_available():
        return ReviewOptions(vendedores=[], categorias=[])
    table, links, _ = _tables()
    commented = table[table["texto"].notna()][["order_id", "nota"]]
    by_seller = commented.merge(links[["order_id", "seller_id"]].drop_duplicates(), on="order_id")
    sellers = by_seller.groupby("seller_id")["nota"].agg(["size", "mean"]).sort_values("size", ascending=False)
    by_cat = commented.merge(links[["order_id", "categoria"]].dropna().drop_duplicates(), on="order_id")
    cats = by_cat.groupby("categoria").size().sort_values(ascending=False)
    return ReviewOptions(
        vendedores=[OpcionVendedor(seller_id=s, resenas_con_comentario=int(row["size"]),
                                   nota_media=round(float(row["mean"]), 2))
                    for s, row in sellers.head(limit).iterrows()],
        categorias=[OpcionCategoria(categoria=c, etiqueta=category_label(c), resenas_con_comentario=int(v))
                    for c, v in cats.items()])
