import { useEffect, useRef, useState, type FormEvent } from "react";
import { api, type SummaryOutput } from "../api/client";
import { EmptyState, ErrorBox, FormField, LoadingState, MetricCard, Notice, PageHeader } from "../components/common";
import Icon from "../components/Icon";
import { useSystem } from "../components/SystemContext";
import { useApiAction } from "../hooks/useApi";

export default function Reviews() {
  const [mode, setMode] = useState<"texto" | "vendedor">("texto");
  const [text, setText] = useState("");
  const [seller, setSeller] = useState("");
  const [sentences, setSentences] = useState("3");
  const [validation, setValidation] = useState("");
  const [copyState, setCopyState] = useState<"idle" | "copying" | "copied" | "error">("idle");
  const copyAttempt = useRef(0);
  const action = useApiAction<SummaryOutput>();
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
    void action.run((signal) => mode === "texto" ? api.summarize(text.trim(), Number(sentences), signal) : api.summarizeSeller(seller.trim(), Number(sentences), signal));
  };
  return <>
    <PageHeader title="Resumen de reseñas" eyebrow="Opiniones de clientes" subtitle="Resume reseñas en español o portugués." />
    <div className="grid items-start gap-4 lg:grid-cols-2">
      <form className="card space-y-3" onSubmit={run} aria-busy={action.loading}>
        <h2 className="section-title">{mode === "texto" ? "Texto original" : "Reseñas del vendedor"}</h2>
        <div className="flex flex-wrap gap-2 rounded-xl bg-slate-100 p-1.5" role="group" aria-label="Origen de las reseñas">{(["texto", "vendedor"] as const).map((value) => <button type="button" key={value} aria-pressed={mode === value} onClick={() => { reset(); setMode(value); }} className={`min-h-11 flex-1 rounded-lg px-3 py-2 text-sm font-semibold ${mode === value ? "bg-white text-brand-700 shadow-sm" : "text-slate-600 hover:bg-white/60"}`}>{value === "texto" ? "Pegar texto" : "Por vendedor"}</button>)}</div>
        {mode === "texto" ? <FormField id="review-text" label="Reseñas" hint="Mínimo 20 caracteres. Usa oraciones completas.">
          <textarea id="review-text" className="input min-h-44 resize-y leading-6" placeholder="Pega aquí las reseñas…" required minLength={20} value={text} aria-describedby="review-text-hint" onChange={(e) => { reset(); setText(e.target.value); }} />
          <p className="mt-1 text-right text-xs text-slate-500">{text.length} caracteres</p>
        </FormField> : <>
          <FormField id="review-seller" label="Identificador del vendedor" hint="Usa el identificador de Olist, no el nombre."><input id="review-seller" className="input font-mono" placeholder="Identificador del vendedor" required value={seller} aria-describedby="review-seller-hint" onChange={(e) => { reset(); setSeller(e.target.value); }} /></FormField>
          {missing && <Notice title="Reseñas no disponibles" tone="warning">Faltan datos para consultar por vendedor. Puedes pegar el texto directamente.</Notice>}
        </>}
        <div className="flex flex-wrap items-end gap-3"><div className="w-44"><FormField id="review-sentences" label="Oraciones del resumen"><input id="review-sentences" className="input" type="number" min={1} max={15} step={1} required value={sentences} onChange={(e) => { reset(); setSentences(e.target.value); }} /></FormField></div><button type="submit" className="btn sm:ml-auto" disabled={action.loading}><Icon name="text" className="h-4 w-4" />{action.loading ? "Resumiendo…" : "Generar resumen"}</button></div>
        <p className="text-xs leading-5 text-slate-500">De 1 a 15 oraciones, según el texto disponible.</p>
        <ErrorBox>{validation || action.error}</ErrorBox>
      </form>
      <section className="card" aria-label="Resultado del resumen" aria-busy={action.loading}>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2"><h2 className="section-title">Resumen</h2>{action.data?.resumen && <button type="button" className="btn-secondary" onClick={() => void copySummary()} disabled={copyState === "copying"}>{copyState === "copying" ? "Copiando…" : "Copiar resumen"}</button>}</div>
        <p role="status" className="text-xs text-emerald-800">{copyState === "copied" ? "Resumen copiado." : ""}</p>
        {copyState === "error" && <ErrorBox>No se pudo copiar. Selecciona el resumen y cópialo manualmente.</ErrorBox>}
        {action.loading ? <LoadingState label="Seleccionando las oraciones relevantes…" /> : action.data ? <div aria-live="polite">
          {action.data.resumen ? <div className="rounded-lg border border-slate-200 bg-slate-50 p-3"><p id="summary-text" className="whitespace-pre-wrap break-words text-sm leading-6 text-slate-800">{action.data.resumen}</p></div> : <EmptyState icon="text" title="Sin oraciones para resumir" description="Prueba con un texto más descriptivo y oraciones completas." />}
          <div className="mt-3 grid gap-3 sm:grid-cols-2"><MetricCard label="Oraciones originales" value={action.data.oraciones_originales} detail="Elegibles tras el procesamiento." icon="text" /><MetricCard label="Oraciones del resumen" value={action.data.oraciones_resumen} detail="Seleccionadas, no generadas." icon="layers" /></div>
          <p className="mt-3 text-xs leading-5 text-slate-500">No analiza sentimiento ni sustituye la lectura completa.</p>
          {mode === "vendedor" && <p className="mt-2 text-xs leading-5 text-slate-500">El servicio no devuelve el texto original de las reseñas del vendedor.</p>}
        </div> : <EmptyState icon="text" title="El resumen aparecerá aquí" description="Introduce las reseñas y genera el resumen." />}
      </section>
    </div>
    <details className="card mt-4"><summary className="min-h-11 py-2.5 text-sm font-semibold text-slate-700">Cómo se elabora el resumen</summary><p className="text-sm leading-5 text-slate-500">TextRank selecciona oraciones del texto y conserva su orden. No requiere entrenamiento. La consulta por vendedor usa el identificador seller_id y los datos de reseñas e ítems de Olist.</p></details>
  </>;
}
