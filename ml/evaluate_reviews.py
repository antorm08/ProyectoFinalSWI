"""Evaluación del componente 5: resumen de opiniones y reclamos (reseñas de Olist).

No hay resúmenes de referencia escritos por personas, así que se evalúa de dos formas reproducibles:

1. Temas: se contrastan con hechos de los pedidos que el texto no conoce (fecha real de entrega frente a la
   estimada y frente a la fecha de la reseña). Si "pedido no recibido" detecta bien, esos pedidos deberían
   estar sin entregar cuando se escribió la reseña mucho más a menudo que el resto de reclamos.
2. Resumen: se comparan métodos sobre los vendedores con más reseñas, con métricas de cobertura y
   redundancia calculadas con una representación (palabras) distinta de la que usa el método (caracteres).

Uso (desde la raíz del repo):  python ml/evaluate_reviews.py
Escribe docs/evaluacion_resumen.md y docs/evaluacion_resumen_metricas.json. No genera artefactos de modelo.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))
from app.services import summary_service as ss  # noqa: E402  (pipeline real de producción)
from common import load  # noqa: E402

DOCS = ROOT / "docs"
N_FRASES = 5
N_VENDEDORES = 60
MIN_RESENAS_GRUPO = 15
SEMILLAS = [0, 1, 2, 3, 4]
UMBRAL_REPRESENTADA = 0.5
METODOS = {
    "Aleatorio (5 semillas)": None,
    "Más frecuentes": "frecuencia",
    "Centroide TF-IDF": ("centroide", None),
    "TextRank clásico": ("textrank", None),
    "TextRank ponderado": ("textrank_ponderado", None),
    "TextRank ponderado + MMR λ=0.5": ("textrank_mmr", 0.5),
    "TextRank ponderado + MMR λ=0.7 sin umbral de paráfrasis": ("textrank_mmr", 0.7, 1.01),
    "TextRank ponderado + MMR λ=0.7 (elegido)": ("textrank_mmr", 0.7),
    "TextRank ponderado + MMR λ=0.9": ("textrank_mmr", 0.9),
}
ELEGIDO = "TextRank ponderado + MMR λ=0.7 (elegido)"

STOP = sorted(set("""
a ao aos as com como da das de do dos e ela ele em entre era essa esse esta este eu foi for isso ja la mais mas me
meu minha muito na nas nao nem no nos o os ou para pela pelo por pra que se sem seu sua sao tambem tem ter um uma
uns voce bem so ate quando estou estava foram fiquei del el en es esta ha hay las lo los mi muy ni pero porque su sus
tu una y ya
""".split()))


# ---- Métricas de resumen ----
def _eval_vectors(pool: pd.DataFrame):
    vec = CountVectorizer(preprocessor=ss.normalize, stop_words=STOP, binary=True,
                          token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
    X = vec.fit_transform(pool["oracion"]).astype(float)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1
    return X, norms


def summary_metrics(pool: pd.DataFrame, chosen: list[int], themes_top: list[str], X, norms) -> dict:
    w = pool["repeticiones"].to_numpy(dtype=float)
    group_vec = np.asarray(X.T @ w).ravel()
    summ_vec = np.asarray(X[chosen].sum(axis=0)).ravel()
    denom = np.linalg.norm(group_vec) * np.linalg.norm(summ_vec)
    cobertura = float(group_vec @ summ_vec / denom) if denom else 0.0
    sim_to_summary = (X @ X[chosen].T).toarray() / norms[:, None] / norms[chosen][None, :]
    representadas = float(w[sim_to_summary.max(axis=1) >= UMBRAL_REPRESENTADA].sum() / w.sum())
    if len(chosen) > 1:
        s = sim_to_summary[chosen]
        redundancia = float(s[np.triu_indices(len(chosen), 1)].mean())
    else:
        redundancia = 0.0
    found = set().union(*(ss.detect_themes(pool.at[i, "oracion"]) for i in chosen)) if chosen else set()
    temas = float(np.mean([t in found for t in themes_top])) if themes_top else np.nan
    palabras = float(np.mean([len(pool.at[i, "oracion"].split()) for i in chosen])) if chosen else 0.0
    return {"cobertura_contenido": cobertura, "resenas_representadas": representadas,
            "cobertura_temas": temas, "redundancia": redundancia, "palabras_por_frase": palabras}


def choose(pool: pd.DataFrame, method, seed: int = 0) -> list[int]:
    if method is None:
        rng = np.random.default_rng(seed)
        return list(rng.choice(len(pool), size=min(N_FRASES, len(pool)), replace=False))
    if method == "frecuencia":
        return list(np.argsort(-pool["repeticiones"].to_numpy(), kind="stable")[:N_FRASES])
    name, lam, *max_sim = method
    return ss.rank_pool(pool, N_FRASES, method=name, lam=lam if lam is not None else ss.MMR_LAMBDA,
                        max_sim=max_sim[0] if max_sim else None)


def top_themes(group: pd.DataFrame, k: int = 3) -> list[str]:
    shares = group[[t for t in ss.TEMAS if t not in ss.TEMAS_GENERICOS]].mean().sort_values(ascending=False)
    return [t for t, v in shares.items() if v >= 0.05][:k]


def compare_methods(table: pd.DataFrame, links: pd.DataFrame) -> tuple[list[dict], dict]:
    commented = table[table["texto"].notna()]
    per_seller = commented.merge(links[["order_id", "seller_id"]].drop_duplicates(), on="order_id")
    rec_counts = per_seller[per_seller["nota"] <= ss.NOTA_RECLAMO].groupby("seller_id").size()
    sellers = rec_counts[rec_counts >= MIN_RESENAS_GRUPO].index
    sellers = per_seller[per_seller["seller_id"].isin(sellers)]["seller_id"].value_counts().index[:N_VENDEDORES]
    rows = []
    for seller in sellers:
        g = per_seller[per_seller["seller_id"] == seller]
        for grupo, part in (("reclamos", g[g["nota"] <= ss.NOTA_RECLAMO]), ("elogios", g[g["nota"] >= ss.NOTA_ELOGIO])):
            pool = ss.sentence_pool(part["texto"])
            if len(pool) <= N_FRASES:
                continue
            X, norms = _eval_vectors(pool)
            themes = top_themes(part)
            for label, method in METODOS.items():
                seeds = SEMILLAS if method is None else [0]
                ms = [summary_metrics(pool, choose(pool, method, s), themes, X, norms) for s in seeds]
                rows.append({"vendedor": seller, "grupo": grupo, "metodo": label,
                             **{k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}})
    df = pd.DataFrame(rows)
    keys = ["cobertura_contenido", "resenas_representadas", "cobertura_temas", "redundancia", "palabras_por_frase"]
    summary = []
    for grupo in ("reclamos", "elogios", "todos"):
        part = df if grupo == "todos" else df[df["grupo"] == grupo]
        for label in METODOS:
            m = part[part["metodo"] == label]
            summary.append({"grupo": grupo, "metodo": label, "n_grupos": int(len(m)),
                            **{k: round(float(m[k].mean()), 4) for k in keys},
                            **{f"{k}_std": round(float(m[k].std()), 4) for k in keys[:4]}})
    # ¿En cuántos grupos el método elegido cubre al menos tanto como TextRank clásico?
    pivot = df.pivot_table(index=["vendedor", "grupo"], columns="metodo", values="resenas_representadas")
    wins = {"vs_textrank_clasico": round(float((pivot[ELEGIDO] >= pivot["TextRank clásico"]).mean()), 4),
            "vs_centroide": round(float((pivot[ELEGIDO] >= pivot["Centroide TF-IDF"]).mean()), 4),
            "vs_aleatorio": round(float((pivot[ELEGIDO] >= pivot["Aleatorio (5 semillas)"]).mean()), 4)}
    return summary, {"vendedores": int(len(sellers)), "grupos": int(len(pivot)), "proporcion_grupos_no_peor": wins}


# ---- Validación de temas con hechos de los pedidos ----
def theme_validation(reviews_raw: pd.DataFrame, table: pd.DataFrame) -> list[dict]:
    orders = load("orders")
    for c in ("order_delivered_customer_date", "order_estimated_delivery_date"):
        orders[c] = pd.to_datetime(orders[c])
    r = reviews_raw.drop_duplicates(["review_id", "order_id"]).reset_index(drop=True)
    data = pd.concat([table.reset_index(drop=True), r[["review_creation_date"]]], axis=1)
    data["review_creation_date"] = pd.to_datetime(data["review_creation_date"])
    data = data.merge(orders[["order_id", "order_status", "order_delivered_customer_date",
                              "order_estimated_delivery_date"]], on="order_id", how="left")
    delivered = data["order_delivered_customer_date"]
    data["retrasado"] = delivered.dt.normalize() > data["order_estimated_delivery_date"].dt.normalize()
    # Sin entregar al escribir la reseña: no hay fecha de entrega o es posterior a la reseña.
    data["sin_entregar_al_opinar"] = delivered.isna() | (delivered.dt.normalize() > data["review_creation_date"])
    data["adelantado"] = delivered.notna() & (delivered.dt.normalize() < data["order_estimated_delivery_date"].dt.normalize())
    commented = data["texto"].notna()
    checks = [
        ("no_recibido", "reclamos", "sin_entregar_al_opinar", "Pedido sin entregar cuando se escribió la reseña"),
        ("plazo_entrega", "reclamos", "retrasado", "Entregado después de la fecha estimada"),
        ("plazo_entrega", "elogios", "adelantado", "Entregado antes de la fecha estimada"),
        ("incompleto", "reclamos", "sin_entregar_al_opinar", "Control: incompleto frente a sin entregar"),
        ("calidad", "reclamos", "retrasado", "Control: calidad frente a retraso"),
    ]
    out = []
    for tema, grupo, hecho, desc in checks:
        mask = commented & ((data["nota"] <= ss.NOTA_RECLAMO) if grupo == "reclamos" else (data["nota"] >= ss.NOTA_ELOGIO))
        part = data[mask]
        con, sin = part[part[tema]], part[~part[tema]]
        p_con, p_sin = float(con[hecho].mean()), float(sin[hecho].mean())
        out.append({"tema": tema, "grupo": grupo, "hecho": desc, "n_con_tema": int(len(con)),
                    "n_sin_tema": int(len(sin)), "tasa_con_tema": round(p_con, 4), "tasa_sin_tema": round(p_sin, 4),
                    "lift": round(p_con / p_sin, 2) if p_sin else None})
    return out


def theme_coverage(table: pd.DataFrame) -> dict:
    commented = table[table["texto"].notna()]
    keys = list(ss.TEMAS)
    res = {}
    for grupo, part in (("reclamos", commented[commented["nota"] <= ss.NOTA_RECLAMO]),
                        ("neutras", commented[commented["nota"] == 3]),
                        ("elogios", commented[commented["nota"] >= ss.NOTA_ELOGIO])):
        res[grupo] = {"resenas": int(len(part)), "con_algun_tema": round(float(part[keys].any(axis=1).mean()), 4),
                      "por_tema": {k: round(float(part[k].mean()), 4) for k in keys}}
    return res


def theme_samples(table: pd.DataFrame, k: int = 4) -> dict[str, list[str]]:
    commented = table[table["texto"].notna() & (table["nota"] <= ss.NOTA_RECLAMO)]
    return {t: commented[commented[t]]["texto"].sample(min(k, int(commented[t].sum())), random_state=42)
            .str.replace("\n", " ").str.slice(0, 140).tolist() for t in ss.TEMAS}


# ---- Ejemplos y latencia ----
def examples_and_latency() -> tuple[list[dict], dict]:
    ss._tables.cache_clear()
    ss.summarize_platform.cache_clear()
    t = time.perf_counter(); ss._tables(); carga = time.perf_counter() - t
    opts = ss.review_options(200)
    t = time.perf_counter(); platform = ss.summarize_platform(N_FRASES); t_plat = time.perf_counter() - t
    big = [v for v in opts.vendedores if v.resenas_con_comentario >= 100]
    worst = min(big, key=lambda v: v.nota_media)
    best = max(big, key=lambda v: v.nota_media)
    times = []
    examples = [platform]
    for v in (worst, best):
        t = time.perf_counter(); examples.append(ss.summarize_seller(v.seller_id, N_FRASES))
        times.append(time.perf_counter() - t)
    t = time.perf_counter(); examples.append(ss.summarize_category(opts.categorias[0].categoria, N_FRASES))
    times.append(time.perf_counter() - t)
    t = time.perf_counter(); ss.summarize("O produto chegou rápido. A qualidade é ótima. Recomendo a loja. " * 20, 3)
    t_texto = time.perf_counter() - t
    lat = {"carga_inicial_s": round(carga, 2), "plataforma_primera_vez_s": round(t_plat, 2),
           "vendedor_o_categoria_max_s": round(max(times), 2), "texto_pegado_s": round(t_texto, 3)}
    return [e.model_dump() for e in examples], lat


# ---- Informe ----
def fmt_pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"


def write_report(report: dict) -> None:
    lines = ["# Evaluación del módulo de resumen de opiniones y reclamos (Olist)", "",
             "Generado con `python ml/evaluate_reviews.py`. Todos los números se recalculan con ese comando.", ""]
    d = report["datos"]
    lines += ["## 1. Datos", "",
              f"- Reseñas: **{d['resenas']:,}**; con comentario: **{d['con_comentario']:,}** "
              f"({fmt_pct(d['con_comentario'] / d['resenas'])}).".replace(",", " "),
              f"- Longitud mediana del comentario: {d['mediana_caracteres']} caracteres; oraciones útiles por "
              f"comentario (≥ {ss.MIN_PALABRAS} palabras): {d['oraciones_por_comentario']:.2f}.",
              f"- Reclamos (nota 1–2): {fmt_pct(d['pct_reclamos'])} · neutras (3): {fmt_pct(d['pct_neutras'])} · "
              f"elogios (4–5): {fmt_pct(d['pct_elogios'])}.",
              f"- Oraciones repetidas literalmente (misma clave normalizada): {fmt_pct(d['pct_oraciones_repetidas'])} "
              "de las apariciones. Por eso el resumen pondera cada frase por el número de reseñas que la contienen.", ""]
    lines += ["## 2. Detección de temas", "", "### 2.1 Cobertura", "",
              "| Grupo | Reseñas | Con algún tema |", "|---|---:|---:|"]
    for g, v in report["temas"]["cobertura"].items():
        lines.append(f"| {g} | {v['resenas']:,} | {fmt_pct(v['con_algun_tema'])} |".replace(",", " "))
    lines += ["", "| Tema | % de reclamos | % de elogios |", "|---|---:|---:|"]
    cov = report["temas"]["cobertura"]
    for k, (label, _) in ss.TEMAS.items():
        lines.append(f"| {label} | {fmt_pct(cov['reclamos']['por_tema'][k])} | {fmt_pct(cov['elogios']['por_tema'][k])} |")
    lines += ["", "### 2.2 Validación con hechos de los pedidos", "",
              "El detector solo lee el texto; las fechas de entrega salen de `olist_orders_dataset.csv`. "
              "*Lift* = tasa con el tema / tasa sin el tema, dentro del mismo grupo de notas (así la nota no influye).", "",
              "| Tema | Grupo | Hecho comprobado | Con tema | Sin tema | Lift |", "|---|---|---|---:|---:|---:|"]
    for v in report["temas"]["validacion"]:
        lines.append(f"| {ss.TEMAS[v['tema']][0]} | {v['grupo']} | {v['hecho']} | {fmt_pct(v['tasa_con_tema'])} "
                     f"(n={v['n_con_tema']}) | {fmt_pct(v['tasa_sin_tema'])} | {v['lift']} |")
    lines += ["", "### 2.3 Muestra para revisión manual (reclamos, semilla 42)", ""]
    for k, samples in report["temas"]["muestras"].items():
        lines.append(f"- **{ss.TEMAS[k][0]}:** " + " · ".join(f"“{s}”" for s in samples))
    c = report["comparacion"]
    lines += ["", "## 3. Comparación de métodos de resumen", "",
              f"{c['contexto']['grupos']} grupos (reclamos y elogios de los {c['contexto']['vendedores']} vendedores con "
              f"más reseñas y al menos {MIN_RESENAS_GRUPO} reclamos comentados); {N_FRASES} frases por resumen.", "",
              "- **Cobertura de contenido:** coseno entre las palabras del resumen y las del grupo (ponderadas por repetición).",
              f"- **Reseñas representadas:** % de oraciones del grupo cuyo vocabulario coincide en coseno ≥ {UMBRAL_REPRESENTADA} "
              "con alguna frase del resumen.",
              "- **Cobertura de temas:** de los 3 temas principales del grupo, cuántos aparecen en el resumen.",
              "- **Redundancia:** similitud media entre las frases del resumen (menor es mejor).",
              "- Las métricas usan palabras; el método usa n-gramas de caracteres, para no evaluarse con su propia representación.", ""]
    for grupo in ("todos", "reclamos", "elogios"):
        lines += [f"### {grupo.capitalize()}", "",
                  "| Método | Cobertura contenido ↑ | Reseñas representadas ↑ | Cobertura temas ↑ | Redundancia ↓ | Palabras/frase |",
                  "|---|---:|---:|---:|---:|---:|"]
        for m in (x for x in c["resumen"] if x["grupo"] == grupo):
            lines.append(f"| {m['metodo']} | {m['cobertura_contenido']:.3f} | {fmt_pct(m['resenas_representadas'])} | "
                         f"{fmt_pct(m['cobertura_temas'])} | {m['redundancia']:.3f} | {m['palabras_por_frase']:.1f} |")
        lines.append("")
    w = c["contexto"]["proporcion_grupos_no_peor"]
    lines += [f"En reseñas representadas, el método elegido iguala o supera a TextRank clásico en el "
              f"{fmt_pct(w['vs_textrank_clasico'])} de los grupos, al centroide en el {fmt_pct(w['vs_centroide'])} "
              f"y al azar en el {fmt_pct(w['vs_aleatorio'])}.", "",
              "### Decisión", "", report["decision"], ""]
    lines += ["### Ejemplo: los 5 reclamos más representativos de toda la plataforma", "",
              "| " + " | ".join(report["contraste"]) + " |", "|" + "---|" * len(report["contraste"])]
    lines += ["| " + " | ".join(col[i] for col in report["contraste"].values()) + " |" for i in range(N_FRASES)]
    lines += ["", "## 4. Ejemplos de resúmenes", ""]
    for e in report["ejemplos"]:
        lines += [f"### {e['etiqueta']}" + (f" (`{e['identificador']}`)" if e["identificador"] else ""), "",
                  f"> {e['resumen_ejecutivo']}", "", "**Reclamos representativos**", ""]
        lines += [f"- {p['texto']}" + (f" (×{p['repeticiones']})" if p["repeticiones"] > 1 else "") for p in e["reclamos"]]
        lines += ["", "**Elogios representativos**", ""]
        lines += [f"- {p['texto']}" + (f" (×{p['repeticiones']})" if p["repeticiones"] > 1 else "") for p in e["elogios"]]
        lines.append("")
    lat = report["latencia"]
    lines += ["## 5. Tiempo de respuesta (CPU, sin GPU)", "",
              f"- Carga y preparación de las reseñas (una vez por proceso): {lat['carga_inicial_s']} s",
              f"- Resumen de toda la plataforma (primera vez; luego queda en caché): {lat['plataforma_primera_vez_s']} s",
              f"- Resumen de un vendedor o categoría: ≤ {lat['vendedor_o_categoria_max_s']} s",
              f"- Texto pegado (60 oraciones): {lat['texto_pegado_s']} s", "",
              "## 6. Limitaciones", "",
              "- Es un resumen **extractivo**: las frases salen tal cual de las reseñas, en portugués. El resumen "
              "ejecutivo en español se arma con plantillas a partir de los indicadores; no traduce ni inventa contenido.",
              "- Los temas se detectan por léxico: es transparente y auditable, pero no entiende negaciones complejas "
              "ni ironía. Los lifts de la sección 2.2 muestran que aun así capturan señales reales.",
              "- Reclamo y elogio se definen por la nota (1–2 y 4–5). Las reseñas de nota 3 cuentan en los totales "
              "pero no en esas dos secciones.",
              "- Un modelo abstractivo (p. ej. mT5 o un LLM) podría redactar el resumen en español, pero requiere GPU o "
              "una API externa. La interfaz del servicio permite sustituirlo sin cambiar el frontend.", ""]
    (DOCS / "evaluacion_resumen.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    reviews_raw = load("order_reviews")
    items = load("order_items")
    products = load("products")
    table, links = ss.build_tables(reviews_raw, items, products)
    commented = table[table["texto"].notna()]
    total_sent, unique_keys = 0, set()
    for text in commented["texto"]:
        sents = ss.split_sentences(text)
        total_sent += len(sents)
        unique_keys.update(ss.sentence_key(s) for s in sents)
    datos = {"resenas": int(len(table)), "con_comentario": int(len(commented)),
             "mediana_caracteres": int(commented["texto"].str.len().median()),
             "oraciones_por_comentario": total_sent / len(commented),
             "pct_oraciones_repetidas": 1 - len(unique_keys) / total_sent,
             "pct_reclamos": float((table["nota"] <= 2).mean()), "pct_neutras": float((table["nota"] == 3).mean()),
             "pct_elogios": float((table["nota"] >= 4).mean())}
    print("Validando temas…")
    temas = {"cobertura": theme_coverage(table), "validacion": theme_validation(reviews_raw, table),
             "muestras": theme_samples(table)}
    print("Comparando métodos de resumen…")
    resumen, contexto = compare_methods(table, links)
    print("Generando ejemplos…")
    ejemplos, latencia = examples_and_latency()
    rec_pool = ss.sentence_pool(commented.loc[commented["nota"] <= ss.NOTA_RECLAMO, "texto"])
    contraste = {label: [rec_pool.at[i, "oracion"] for i in ss.rank_pool(rec_pool, N_FRASES, method=m)]
                 for label, m in (("TextRank clásico", "textrank"), ("Centroide TF-IDF", "centroide"),
                                  ("TextRank ponderado + MMR (elegido)", "textrank_mmr"))}
    best = {m["metodo"]: m for m in resumen if m["grupo"] == "todos"}
    e, tr, ce = best[ELEGIDO], best["TextRank clásico"], best["Centroide TF-IDF"]
    l5 = best["TextRank ponderado + MMR λ=0.5"]
    nu = best["TextRank ponderado + MMR λ=0.7 sin umbral de paráfrasis"]
    decision = (
        f"Se elige **TextRank ponderado por frecuencia + MMR (λ = {ss.MMR_LAMBDA})** con n-gramas de caracteres. "
        f"Frente a TextRank clásico cubre más contenido ({e['cobertura_contenido']:.3f} frente a "
        f"{tr['cobertura_contenido']:.3f}), representa más reseñas ({fmt_pct(e['resenas_representadas'])} frente a "
        f"{fmt_pct(tr['resenas_representadas'])}), recoge más temas principales ({fmt_pct(e['cobertura_temas'])} frente "
        f"a {fmt_pct(tr['cobertura_temas'])}) y repite menos ({e['redundancia']:.3f} frente a {tr['redundancia']:.3f}; "
        f"centroide: {ce['redundancia']:.3f}). Sin MMR, TextRank y el centroide eligen variantes de la misma frase "
        f"(“não recebi o produto” escrito de cinco formas). El coste es que las frases elegidas son algo más largas "
        f"({e['palabras_por_frase']:.1f} frente a {tr['palabras_por_frase']:.1f} palabras). "
        f"λ = 0.5 diversifica aún más (temas {fmt_pct(l5['cobertura_temas'])}, redundancia {l5['redundancia']:.3f}) "
        f"pero elige frases más largas ({l5['palabras_por_frase']:.1f} palabras) y cubre menos contenido "
        f"({l5['cobertura_contenido']:.3f}); λ = 0.9 se acerca a TextRank sin MMR. Ponderar por frecuencia casi no "
        f"cambia las métricas, porque el filtro de candidatas ya usa la frecuencia, pero hace que la frase que "
        f"comparten más clientes salga primero. El umbral de paráfrasis (coseno > {ss.MAX_SIMILITUD}) mejora poco "
        f"frente a no usarlo (redundancia {nu['redundancia']:.3f} → {e['redundancia']:.3f}, temas "
        f"{fmt_pct(nu['cobertura_temas'])} → {fmt_pct(e['cobertura_temas'])}), pero evita que en vendedores con pocas "
        f"reseñas salgan dos versiones de la misma frase. Ningún método supera al azar en redundancia: el azar elige frases "
        f"sin relación entre sí, a costa de cubrir mucho menos ({best['Aleatorio (5 semillas)']['cobertura_contenido']:.3f}).")
    report = {"datos": datos, "temas": temas, "comparacion": {"resumen": resumen, "contexto": contexto},
              "decision": decision, "contraste": contraste, "ejemplos": ejemplos, "latencia": latencia,
              "parametros": {"n_frases": N_FRASES, "mmr_lambda": ss.MMR_LAMBDA, "max_candidatas": ss.MAX_CANDIDATAS,
                             "min_palabras": ss.MIN_PALABRAS, "umbral_representada": UMBRAL_REPRESENTADA}}
    DOCS.mkdir(exist_ok=True)
    (DOCS / "evaluacion_resumen_metricas.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "ejemplos"}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    write_report(report)
    print(json.dumps({"validacion": temas["validacion"], "todos": [m for m in resumen if m["grupo"] == "todos"],
                      "contexto": contexto, "latencia": latencia}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
