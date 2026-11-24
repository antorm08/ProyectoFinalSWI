import { useEffect, useRef, useState, type FormEvent } from "react";
import { api, type FraseRepresentativa, type ReviewInsights, type SummaryOutput, type TemaDetectado } from "../api/client";
import { EmptyState, ErrorBox, FormField, LoadingState, MetricCard, Notice, PageHeader, StatusBadge } from "../components/common";
import Icon from "../components/Icon";
import { useSystem } from "../components/SystemContext";
import { useApiAction, useApiQuery } from "../hooks/useApi";
import { number, percent } from "../utils/format";

type Mode = "texto" | "vendedor" | "categoria" | "plataforma";
const MODES: [Mode, string][] = [["texto", "Pegar texto"], ["vendedor", "Por vendedor"], ["categoria", "Por categoría"], ["plataforma", "Toda la plataforma"]];
const TITLES: Record<Mode, string> = { texto: "Texto original", vendedor: "Reseñas del vendedor", categoria: "Reseñas de una categoría", plataforma: "Todas las reseñas" };
const AMBITO = { vendedor: "Vendedor", categoria: "Categoría", plataforma: "Plataforma" } as const;
const isInsights = (data: SummaryOutput | ReviewInsights): data is ReviewInsights => "ambito" in data;
const loadOptions = (signal: AbortSignal) => api.reviewOptions(signal);

interface BarRow { key: string; label: string; value: number; detail: string; reference?: number | null }

/** Barras horizontales de una sola serie; la marca vertical indica la media de la plataforma. */
function Bars({ rows, label, referenceLabel }: { rows: BarRow[]; label: string; referenceLabel?: string }) {
  const max = Math.max(0.1, ...rows.map((r) => Math.max(r.value, r.reference ?? 0)));
  const domain = Math.min(1, Math.ceil(max * 10) / 10);
  const hasReference = rows.some((r) => r.reference != null);
  return <div>
    <ul aria-label={label} className="space-y-2.5">{rows.map((row) => <li key={row.key} title={row.detail} className="group">
      <div className="flex items-baseline justify-between gap-3 text-sm"><span className="min-w-0 text-slate-700">{row.label}</span><span className="font-semibold tabular-nums text-slate-800">{percent(row.value)}</span></div>
      <div className="relative mt-1 h-2 rounded-full bg-slate-100">
        <div className="h-2 rounded-full bg-brand-600 transition-opacity group-hover:opacity-80" style={{ width: `${Math.max(1, (row.value / domain) * 100)}%` }} />
        {row.reference != null && <span aria-hidden="true" className="absolute -top-1 h-4 w-0.5 rounded-full bg-slate-800" style={{ left: `calc(${(row.reference / domain) * 100}% - 1px)` }} />}
      </div>
      <p className="sr-only">{row.detail}</p>
    </li>)}</ul>
    {hasReference && referenceLabel && <p className="mt-3 flex items-center gap-2 text-xs text-slate-500" aria-hidden="true"><span className="inline-block h-2 w-5 rounded-full bg-brand-600" />Este conjunto<span className="ml-2 inline-block h-3.5 w-0.5 rounded-full bg-slate-800" />{referenceLabel}</p>}
  </div>;
}

function themeRows(temas: TemaDetectado[], what: string, withReference: boolean): BarRow[] {
  return temas.map((t) => ({
    key: t.tema, label: t.etiqueta, value: t.porcentaje, reference: withReference ? t.porcentaje_plataforma : null,
    detail: `${t.etiqueta}: ${percent(t.porcentaje)} de ${what} (${number(t.menciones)} menciones)${withReference && t.porcentaje_plataforma != null ? `; media de la plataforma: ${percent(t.porcentaje_plataforma)}` : ""}.`,
  }));
}

function Phrases({ title, items, labels, empty }: { title: string; items: FraseRepresentativa[]; labels: Record<string, string>; empty: string }) {
  return <section aria-label={title}>
    <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
    {items.length ? <ol className="mt-2 space-y-2">{items.map((item, i) => <li key={`${i}-${item.texto}`} className="rounded-lg border border-slate-200 bg-slate-50 p-2.5">
      <p lang="pt" className="break-words text-sm leading-6 text-slate-800">{item.texto}</p>
      <div className="mt-1.5 flex flex-wrap gap-1.5">
        <span className="rounded-full bg-white px-2 py-0.5 text-xs font-medium text-slate-700 ring-1 ring-inset ring-slate-200">{item.repeticiones > 1 ? `${number(item.repeticiones)} reseñas la repiten` : "1 reseña"}</span>
        {item.temas.map((t) => <span key={t} className="rounded-full bg-brand-50 px-2 py-0.5 text-xs font-medium text-brand-700">{labels[t] ?? t}</span>)}
      </div>
    </li>)}</ol> : <p className="mt-2 text-sm text-slate-500">{empty}</p>}
  </section>;
}

function InsightsView({ data }: { data: ReviewInsights }) {
  const labels = Object.fromEntries([...(data.temas ?? []), ...data.temas_reclamos, ...data.temas_elogios].map((t) => [t.tema, t.etiqueta]));
  const compare = data.ambito !== "plataforma";
  const ratings: BarRow[] = ["5", "4", "3", "2", "1"].map((k) => {
    const count = data.distribucion_notas[k] ?? 0;
    const value = data.resenas_totales ? count / data.resenas_totales : 0;
    return { key: k, label: `${k} ${k === "1" ? "estrella" : "estrellas"}`, value, detail: `${k} ${k === "1" ? "estrella" : "estrellas"}: ${number(count)} reseñas (${percent(value)}).` };
  });
  return <div aria-live="polite" className="space-y-4">
    <div><div className="flex flex-wrap items-center gap-2"><StatusBadge tone="info">{AMBITO[data.ambito]}</StatusBadge><h3 className="min-w-0 break-words text-base font-semibold">{data.etiqueta}</h3></div>
      {data.ambito === "vendedor" && data.identificador && <p className="mt-1 break-all font-mono text-xs text-slate-500">{data.identificador}</p>}</div>
    {data.advertencias.map((a) => <Notice key={a} title="Ten en cuenta" tone="warning">{a}</Notice>)}
    <div className="rounded-lg border border-brand-100 bg-brand-50 p-3"><p className="mb-1 text-xs font-semibold uppercase tracking-wide text-brand-700">Resumen ejecutivo</p><p id="insights-summary" className="text-sm leading-6 text-slate-800">{data.resumen_ejecutivo}</p></div>
    <div className="grid gap-3 sm:grid-cols-2">
      <MetricCard label="Reseñas analizadas" value={number(data.resenas_totales)} detail={`${number(data.resenas_con_comentario)} con comentario.`} icon="text" />
      <MetricCard label="Nota media" value={`${number(data.nota_media, 2)} / 5`} detail="Sobre todas las reseñas, con o sin comentario." icon="chart" />
      <MetricCard label="Reclamos" value={percent(data.pct_reclamos)} detail={compare ? `Notas 1–2. Media de la plataforma: ${percent(data.pct_reclamos_plataforma)}.` : "Reseñas con nota 1–2."} icon="alert" />
      <MetricCard label="Elogios" value={percent(data.pct_elogios)} detail="Reseñas con nota 4–5." icon="check" />
    </div>
    <section aria-label="Distribución de notas"><h3 className="mb-2 text-sm font-semibold text-slate-800">Distribución de notas</h3><Bars rows={ratings} label="Porcentaje de reseñas por nota" /></section>
    <Phrases title="Reclamos representativos" items={data.reclamos} labels={labels} empty="No hay reclamos con comentario." />
    {data.temas_reclamos.length > 0 && <section aria-label="Temas de los reclamos"><h3 className="mb-1 text-sm font-semibold text-slate-800">Temas de los reclamos</h3><p className="mb-2 text-xs leading-5 text-slate-500">% de reclamos con comentario que mencionan cada tema. Una reseña puede tener varios.</p><Bars rows={themeRows(data.temas_reclamos, "los reclamos", compare)} label="Temas mencionados en los reclamos" referenceLabel="Media de la plataforma" /></section>}
    <Phrases title="Lo más valorado" items={data.elogios} labels={labels} empty="No hay elogios con comentario." />
    {data.temas_elogios.length > 0 && <section aria-label="Temas de los elogios"><h3 className="mb-2 text-sm font-semibold text-slate-800">Temas de los elogios</h3><Bars rows={themeRows(data.temas_elogios, "los elogios", false)} label="Temas mencionados en los elogios" /></section>}
    <p className="text-xs leading-5 text-slate-500">Las frases se citan tal cual, en portugués (idioma de las reseñas de Olist). Se eligen, no se generan.</p>
  </div>;
}

export default function Reviews() {
  const [mode, setMode] = useState<Mode>("texto");
  const [text, setText] = useState("");
  const [seller, setSeller] = useState("");
  const [category, setCategory] = useState("");
  const [sentences, setSentences] = useState("3");
  const [validation, setValidation] = useState("");
  const [copyState, setCopyState] = useState<"idle" | "copying" | "copied" | "error">("idle");
  const copyAttempt = useRef(0);
  const action = useApiAction<SummaryOutput | ReviewInsights>();
  const options = useApiQuery(loadOptions);
  const system = useSystem();
  const missing = system.data && (!system.data.datos_olist.reviews || !system.data.datos_olist.items);
  useEffect(() => () => { copyAttempt.current += 1; }, []);
  const clearCopy = () => { copyAttempt.current += 1; setCopyState("idle"); };
  const reset = () => { action.reset(); setValidation(""); clearCopy(); };
  const copySummary = async () => {
    if (!action.data?.resumen) return;
    const attempt = ++copyAttempt.current;
    setCopyState("copying");
    try {
      await navigator.clipboard.writeText(action.data.resumen);
      if (attempt === copyAttempt.current) setCopyState("copied");
    } catch {
      if (attempt === copyAttempt.current) setCopyState("error");
    }
  };
  const run = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setValidation("");
    clearCopy();
    if (mode === "texto" && text.trim().length < 20) { setValidation("Introduce al menos 20 caracteres de texto, sin contar espacios al inicio o al final."); return; }
    if (mode === "vendedor" && !seller.trim()) { setValidation("Introduce el identificador del vendedor."); return; }
    if (mode === "categoria" && !category) { setValidation("Selecciona una categoría."); return; }
    const n = Number(sentences);
    void action.run((signal) => {
      if (mode === "texto") return api.summarize(text.trim(), n, signal);
      if (mode === "vendedor") return api.summarizeSeller(seller.trim(), n, signal);
      if (mode === "categoria") return api.summarizeCategory(category, n, signal);
      return api.reviewsOverview(n, signal);
    });
  };
  const data = action.data;
  return <>
    <PageHeader title="Resumen de reseñas" eyebrow="Opiniones de clientes" subtitle="Resume reclamos y opiniones por vendedor, por categoría o de un texto pegado." />
    <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
      <form className="card space-y-3" onSubmit={run} aria-busy={action.loading}>
        <h2 className="section-title">{TITLES[mode]}</h2>
        <div className="grid grid-cols-2 gap-2 rounded-xl bg-slate-100 p-1.5" role="group" aria-label="Origen de las reseñas">{MODES.map(([value, label]) => <button type="button" key={value} aria-pressed={mode === value} onClick={() => { reset(); setMode(value); }} className={`min-h-11 rounded-lg px-3 py-2 text-sm font-semibold ${mode === value ? "bg-white text-brand-700 shadow-sm" : "text-slate-600 hover:bg-white/60"}`}>{label}</button>)}</div>
        {mode === "texto" && <FormField id="review-text" label="Reseñas" hint="Mínimo 20 caracteres. Usa oraciones completas.">
          <textarea id="review-text" className="input min-h-44 resize-y leading-6" placeholder="Pega aquí las reseñas…" required minLength={20} value={text} aria-describedby="review-text-hint" onChange={(e) => { reset(); setText(e.target.value); }} />
          <p className="mt-1 text-right text-xs text-slate-500">{text.length} caracteres</p>
        </FormField>}
        {mode === "vendedor" && <>
          <FormField id="review-seller" label="Identificador del vendedor" hint="Escribe o elige uno de los vendedores con más reseñas comentadas.">
            <input id="review-seller" className="input font-mono" placeholder="Identificador del vendedor" required value={seller} list="review-seller-options" autoComplete="off" aria-describedby="review-seller-hint" onChange={(e) => { reset(); setSeller(e.target.value); }} />
            <datalist id="review-seller-options">{options.data?.vendedores.map((v) => <option key={v.seller_id} value={v.seller_id}>{`${number(v.resenas_con_comentario)} reseñas · nota ${number(v.nota_media, 2)}`}</option>)}</datalist>
          </FormField>
          {missing && <Notice title="Reseñas no disponibles" tone="warning">Faltan datos para consultar por vendedor. Puedes pegar el texto directamente.</Notice>}
        </>}
        {mode === "categoria" && <>
          <FormField id="review-category" label="Categoría" hint="Ordenadas por número de reseñas comentadas.">
            <select id="review-category" className="input" required value={category} aria-describedby="review-category-hint" disabled={!options.data?.categorias.length} onChange={(e) => { reset(); setCategory(e.target.value); }}>
              <option value="">{options.loading ? "Cargando categorías…" : "Selecciona una categoría"}</option>
              {options.data?.categorias.map((c) => <option key={c.categoria} value={c.categoria}>{`${c.etiqueta} (${number(c.resenas_con_comentario)})`}</option>)}
            </select>
          </FormField>
          {!options.loading && !options.data?.categorias.length && <Notice title="Categorías no disponibles" tone="warning">Faltan los datos de reseñas de Olist. Puedes pegar el texto directamente.</Notice>}
        </>}
        {mode === "plataforma" && <Notice title="Análisis de toda la plataforma">Reúne todas las reseñas con comentario de Olist. La primera consulta puede tardar unos segundos.</Notice>}
        <div className="flex flex-wrap items-end gap-3"><div className="w-44"><FormField id="review-sentences" label="Oraciones del resumen"><input id="review-sentences" className="input" type="number" min={1} max={15} step={1} required value={sentences} onChange={(e) => { reset(); setSentences(e.target.value); }} /></FormField></div><button type="submit" className="btn sm:ml-auto" disabled={action.loading}><Icon name="text" className="h-4 w-4" />{action.loading ? "Resumiendo…" : "Generar resumen"}</button></div>
        <p className="text-xs leading-5 text-slate-500">{mode === "texto" ? "De 1 a 15 oraciones, según el texto disponible." : "De 1 a 15 frases por sección: reclamos y elogios."}</p>
        <ErrorBox>{validation || action.error}</ErrorBox>
      </form>
      <section className="card min-w-0" aria-label="Resultado del resumen" aria-busy={action.loading}>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2"><h2 className="section-title">Resumen</h2>{data?.resumen && <button type="button" className="btn-secondary" onClick={() => void copySummary()} disabled={copyState === "copying"}>{copyState === "copying" ? "Copiando…" : "Copiar resumen"}</button>}</div>
        <p role="status" className="text-xs text-emerald-800">{copyState === "copied" ? "Resumen copiado." : ""}</p>
        {copyState === "error" && <ErrorBox>No se pudo copiar. Selecciona el resumen y cópialo manualmente.</ErrorBox>}
        {action.loading ? <LoadingState label="Seleccionando las oraciones relevantes…" /> : data ? (isInsights(data) ? <InsightsView data={data} /> : <div aria-live="polite">
          {data.resumen ? <div className="rounded-lg border border-slate-200 bg-slate-50 p-3"><p id="summary-text" className="whitespace-pre-wrap break-words text-sm leading-6 text-slate-800">{data.resumen}</p></div> : <EmptyState icon="text" title="Sin oraciones para resumir" description="Prueba con un texto más descriptivo y oraciones completas." />}
          <div className="mt-3 grid gap-3 sm:grid-cols-2"><MetricCard label="Oraciones originales" value={data.oraciones_originales} detail="Elegibles tras el procesamiento." icon="text" /><MetricCard label="Oraciones del resumen" value={data.oraciones_resumen} detail="Seleccionadas, no generadas." icon="layers" /></div>
          {data.temas && data.temas.length > 0 && <div className="mt-3"><h3 className="text-sm font-semibold text-slate-800">Temas detectados</h3><ul className="mt-2 flex flex-wrap gap-1.5" aria-label="Temas detectados en el texto">{data.temas.map((t) => <li key={t.tema} className="rounded-full bg-brand-50 px-2 py-0.5 text-xs font-medium text-brand-700">{t.etiqueta} · {number(t.menciones)}</li>)}</ul></div>}
          <p className="mt-3 text-xs leading-5 text-slate-500">No sustituye la lectura completa. Para separar reclamos de elogios, consulta por vendedor o categoría.</p>
        </div>) : <EmptyState icon="text" title="El resumen aparecerá aquí" description={mode === "texto" ? "Introduce las reseñas y genera el resumen." : "Elige qué reseñas analizar y genera el resumen."} />}
      </section>
    </div>
    <details className="card mt-4"><summary className="min-h-11 py-2.5 text-sm font-semibold text-slate-700">Cómo se elabora el resumen</summary><div className="space-y-2 text-sm leading-5 text-slate-500">
      <p>Las reseñas se separan por nota: reclamos (1–2) y elogios (4–5). En cada grupo, TextRank ponderado por frecuencia elige las frases más representativas y MMR evita repetir la misma idea. El número junto a cada frase indica cuántas reseñas la contienen.</p>
      <p>Los temas (plazo de entrega, pedido no recibido, calidad, atención…) se detectan con un léxico en portugués y español y se comparan con la media de la plataforma. El resumen ejecutivo se arma a partir de esos indicadores, sin inventar contenido. Evaluación completa en docs/evaluacion_resumen.md.</p>
    </div></details>
  </>;
}
