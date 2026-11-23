import { useState, type FormEvent } from "react";
import { api, type DeliveryDurationOutput } from "../api/client";
import { EmptyState, ErrorBox, FormField, LoadingState, Notice, PageHeader, StatusBadge } from "../components/common";
import Icon from "../components/Icon";
import { useApiAction } from "../hooks/useApi";
import { date, number } from "../utils/format";

const ESTADOS = ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT","PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"];
const DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"];
const MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"];
const DEFAULT_FORM = {
  precio: "120", flete: "20", peso_g: "1500", n_items: "1",
  estado_cliente: "RJ", estado_vendedor: "SP", fecha_compra: "", mes: "", dia_semana: "",
};
type FormKey = keyof typeof DEFAULT_FORM;

export default function Delay() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const action = useApiAction<DeliveryDurationOutput>();
  const set = (key: FormKey, value: string) => {
    action.reset();
    setForm((old) => ({ ...old, [key]: value }));
  };
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const order = {
      precio: Number(form.precio), flete: Number(form.flete), peso_g: Number(form.peso_g), n_items: Number(form.n_items),
      estado_cliente: form.estado_cliente, estado_vendedor: form.estado_vendedor,
    };
    const purchase = form.fecha_compra ? { fecha_compra: form.fecha_compra } : {
      mes: Number(form.mes), dia_semana: Number(form.dia_semana),
    };
    void action.run((signal) => api.predictDeliveryDuration({ ...order, ...purchase }, signal));
  };
  const num = (key: FormKey, label: string, min = 0, step = "any", hint?: string) => <FormField id={`delay-${key}`} label={label} hint={hint}>
    <input id={`delay-${key}`} className="input" type="number" min={min} step={step} required value={form[key]}
      aria-describedby={hint ? `delay-${key}-hint` : undefined} onChange={(e) => set(key, e.target.value)} />
  </FormField>;
  const select = (key: FormKey, label: string, choices: { value: string | number; label: string }[], placeholder?: string) =>
    <FormField id={`delay-${key}`} label={label}>
      <select id={`delay-${key}`} className="input" required value={form[key]} onChange={(e) => set(key, e.target.value)}>
        {placeholder && <option value="" disabled>{placeholder}</option>}
        {choices.map(({ value, label: optionLabel }) => <option key={value} value={value}>{optionLabel}</option>)}
      </select>
    </FormField>;
  return <>
    <PageHeader title="Tiempo de entrega" eyebrow="Predicción logística" subtitle="Estima cuántos días transcurrirán desde la compra hasta la entrega de un pedido." />
    <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
      <form onSubmit={submit} className="card space-y-4" aria-busy={action.loading}>
        <div><h2 className="section-title">Características del pedido</h2><p className="mt-1 text-sm leading-5 text-slate-600">Sustituye los valores de ejemplo por los de tu pedido.</p></div>
        <fieldset>
          <legend className="mb-2 text-sm font-semibold text-slate-800">Importes y carga</legend>
          <div className="grid gap-3 sm:grid-cols-2">
            {num("precio", "Precio total (BRL)")}{num("flete", "Flete (BRL)")}
            {num("peso_g", "Peso total (g)", 0, "any", "Introduce 0 si no conoces el peso total.")}
            {num("n_items", "Cantidad de artículos", 1, "1")}
          </div>
        </fieldset>
        <fieldset className="border-t border-slate-100 pt-3">
          <legend className="mb-2 pr-3 text-sm font-semibold text-slate-800">Ruta en Brasil</legend>
          <div className="grid gap-3 sm:grid-cols-2">
            {select("estado_vendedor", "Estado de origen", ESTADOS.map((value) => ({ value, label: value })))}
            {select("estado_cliente", "Estado de destino", ESTADOS.map((value) => ({ value, label: value })))}
          </div>
          <p className="mt-2 text-xs leading-5 text-slate-600">Si hay varios orígenes, usa el estado con mayor valor total de artículos; en empate, el de más artículos y luego el primero por sigla.</p>
        </fieldset>
        <fieldset className="border-t border-slate-100 pt-3">
          <legend className="mb-2 pr-3 text-sm font-semibold text-slate-800">Compra</legend>
          <FormField id="delay-fecha_compra" label="Fecha de compra (opcional)"
            hint="Si no conoces la fecha exacta, indica el mes y el día de la semana. Solo se mostrará la duración.">
            <input id="delay-fecha_compra" className="input min-w-0" type="date" value={form.fecha_compra}
              aria-describedby="delay-fecha_compra-hint" onChange={(e) => set("fecha_compra", e.target.value)} />
          </FormField>
          {!form.fecha_compra && <div className="mt-3 grid gap-3 sm:grid-cols-2">
            {select("mes", "Mes de compra", MESES.map((label, i) => ({ value: i + 1, label })), "Selecciona el mes")}
            {select("dia_semana", "Día de la semana de compra", DIAS.map((label, i) => ({ value: i, label })), "Selecciona el día")}
          </div>}
        </fieldset>
        <button type="submit" className="btn w-full sm:w-auto" disabled={action.loading}>
          <Icon name="shield" className="h-4 w-4" />{action.loading ? "Estimando entrega…" : "Estimar tiempo de entrega"}
        </button>
      </form>
      <section className="card" aria-label="Resultado de tiempo de entrega" aria-busy={action.loading}>
        <h2 className="section-title mb-3">Resultado</h2>
        <ErrorBox>{action.error}</ErrorBox>
        {action.loading ? <LoadingState label="Calculando la duración…" /> : action.data ? <div className="space-y-4" aria-live="polite">
          <div>
            <StatusBadge tone="warning">Modelo experimental</StatusBadge>
            <p className="mt-1 break-words text-sm text-slate-600">{action.data.modelo}</p>
          </div>
          <dl className="space-y-3">
            <div><dt className="text-sm text-slate-600">Tiempo estimado desde la compra</dt>
              <dd className="text-4xl font-bold tracking-tight text-brand-700 tabular-nums">{number(action.data.dias_entrega_estimados, 2)} días</dd></div>
            <div><dt className="text-sm text-slate-600">Fecha estimada de llegada</dt>
              <dd className="font-semibold">{action.data.fecha_llegada_estimada
                ? <time dateTime={action.data.fecha_llegada_estimada}>{date(action.data.fecha_llegada_estimada)}</time>
                : "No disponible sin fecha de compra"}</dd></div>
          </dl>
          {action.data.fecha_llegada_estimada && <p className="text-sm text-slate-600">Fecha aproximada, sin hora estimada.</p>}
          <div className="border-t border-slate-100 pt-3 text-sm leading-5 text-slate-600">
            <p><strong>Error absoluto medio (MAE):</strong> {number(action.data.mae_desarrollo_dias, 2)} días en desarrollo y {number(action.data.mae_agosto_retrospectivo_dias, 2)} días en agosto de 2018 (retrospectivo).</p>
            <p className="mt-1">Son promedios históricos, no un intervalo de confianza para este pedido.</p>
          </div>
          <Notice title="Menor fiabilidad en entregas prolongadas" tone="warning">
            El modelo puede subestimar estas duraciones. La estimación no es una promesa de entrega ni una probabilidad de retraso.
          </Notice>
        </div> : !action.error && <EmptyState icon="shield" title="Listo para estimar" description="Completa las características del pedido para consultar su duración estimada." />}
      </section>
    </div>
  </>;
}
