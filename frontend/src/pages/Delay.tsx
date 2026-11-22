import { useState, type FormEvent } from "react";
import { api, type DelayInput, type DelayOutput } from "../api/client";
import { EmptyState, ErrorBox, FormField, LoadingState, Notice, PageHeader } from "../components/common";
import Icon from "../components/Icon";
import { useApiAction } from "../hooks/useApi";
import { percent } from "../utils/format";

const ESTADOS = ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT","PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"];
const DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"];
const MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"];
const riskColors: Record<string, string> = { bajo: "text-emerald-700", medio: "text-amber-800", alto: "text-red-700" };
const riskBars: Record<string, string> = { bajo: "bg-emerald-600", medio: "bg-amber-500", alto: "bg-red-600" };

export default function Delay() {
  const [form, setForm] = useState<DelayInput>({ precio: 120, flete: 20, peso_g: 1500, n_items: 1, dias_estimados: 20, estado_cliente: "RJ", estado_vendedor: "SP", mes: 11, dia_semana: 0 });
  const action = useApiAction<DelayOutput>();
  const set = (key: keyof DelayInput, value: string) => {
    action.reset();
    setForm((old) => ({ ...old, [key]: key.startsWith("estado") ? value : value === "" ? "" : Number(value) }));
  };
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void action.run((signal) => api.predictDelay(form, signal));
  };
  const num = (key: keyof DelayInput, label: string, min = 0, step = "any") => <FormField id={`delay-${key}`} label={label}>
    <input id={`delay-${key}`} className="input" type="number" min={min} step={step} required value={form[key]} onChange={(e) => set(key, e.target.value)} />
  </FormField>;
  const select = (key: keyof DelayInput, label: string, options: { value: string | number; label: string }[]) => <FormField id={`delay-${key}`} label={label}>
    <select id={`delay-${key}`} className="input" value={form[key]} onChange={(e) => set(key, e.target.value)}>{options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select>
  </FormField>;
  return <>
    <PageHeader title="Riesgo de entregas" eyebrow="Predicción logística" subtitle="Estima el riesgo de incumplir el plazo de entrega de un pedido." />
    <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
      <form onSubmit={submit} className="card space-y-4" aria-busy={action.loading}>
        <div><h2 className="section-title">Evaluar un pedido</h2><p className="mt-1 text-sm leading-5 text-slate-500">Sustituye los valores de ejemplo por los de tu pedido.</p></div>
        <fieldset><legend className="mb-2 text-sm font-semibold text-slate-800">Importes y carga</legend><div className="grid gap-3 sm:grid-cols-2">{num("precio", "Precio total (BRL)")}{num("flete", "Flete (BRL)")}{num("peso_g", "Peso (g)")}{num("n_items", "Cantidad de artículos", 1, "1")}</div></fieldset>
        <fieldset className="border-t border-slate-100 pt-3"><legend className="mb-2 pr-3 text-sm font-semibold text-slate-800">Entrega</legend><div className="grid gap-3 sm:grid-cols-2">
          {select("estado_vendedor", "Estado de origen", ESTADOS.map((value) => ({ value, label: value })))}
          {select("estado_cliente", "Estado de destino", ESTADOS.map((value) => ({ value, label: value })))}
          {num("dias_estimados", "Plazo prometido (días)", 0, "1")}
        </div><p className="mt-2 text-xs text-slate-500">Las siglas corresponden a estados de Brasil.</p></fieldset>
        <fieldset className="border-t border-slate-100 pt-3"><legend className="mb-2 pr-3 text-sm font-semibold text-slate-800">Compra</legend><div className="grid gap-3 sm:grid-cols-2">
          {select("mes", "Mes de compra", MESES.map((label, i) => ({ value: i + 1, label })))}
          {select("dia_semana", "Día de compra", DIAS.map((label, i) => ({ value: i, label })))}
        </div></fieldset>
        <button type="submit" className="btn w-full sm:w-auto" disabled={action.loading}><Icon name="shield" className="h-4 w-4" />{action.loading ? "Evaluando pedido…" : "Evaluar riesgo de entrega"}</button>
      </form>
      <section className="card" aria-label="Resultado del riesgo" aria-busy={action.loading}>
        <h2 className="section-title mb-3">Resultado</h2>
        <ErrorBox>{action.error}</ErrorBox>
        {action.loading ? <LoadingState label="Calculando el riesgo…" /> : action.data ? <div aria-live="polite">
          {!action.data.modelo_entrenado && <Notice title="Estimación orientativa" tone="warning">Sin modelo entrenado: se aplican reglas de referencia, no una predicción validada.</Notice>}
          <p className="mt-4 text-sm text-slate-500">Probabilidad de retraso</p><p className={`mt-1 text-4xl font-bold tracking-tight tabular-nums ${riskColors[action.data.nivel_riesgo] ?? "text-slate-800"}`}>{percent(action.data.probabilidad_retraso)}</p>
          <p className="mt-2 font-semibold">Riesgo {action.data.nivel_riesgo}</p>
          <div className="relative mt-4 h-2 overflow-hidden rounded-full bg-slate-100" aria-hidden="true"><div className={`h-full rounded-full ${riskBars[action.data.nivel_riesgo] ?? "bg-brand-600"}`} style={{ width: `${Math.max(0, Math.min(100, action.data.probabilidad_retraso * 100))}%` }} /></div>
          <div className="mt-2 flex justify-between text-xs text-slate-500"><span>0 %</span><span>100 %</span></div>
          <details className="mt-3 border-t border-slate-100"><summary className="min-h-11 py-3 text-sm font-semibold text-slate-700">Cómo se calcula</summary><p className="text-xs leading-5 text-slate-500">{action.data.modelo_entrenado ? "Gradient Boosting entrenado; métricas de precisión no disponibles en esta pantalla." : "Reglas orientativas mientras el modelo está pendiente."} Bajo: menos del 20 % · medio: desde el 20 % hasta menos del 50 % · alto: desde el 50 %.</p></details>
        </div> : !action.error && <EmptyState icon="shield" title="Listo para evaluar" description="Completa el formulario y evalúa el riesgo." />}
      </section>
    </div>
  </>;
}
