import { useState, type FormEvent } from "react";
import { api, type DeliveryDurationOutput } from "../api/client";
import { EmptyState, ErrorBox, FormField, LoadingState, PageHeader, StatusBadge } from "../components/common";
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

function FieldHelp({ id, label, children }: { id: string; label: string; children: string }) {
  return <details>
    <summary className="min-h-11 w-fit cursor-pointer py-3 text-sm font-medium text-brand-700 lg:min-h-8 lg:py-1.5">{label}</summary>
    <p id={id} className="pb-2 text-sm leading-5 text-slate-600">{children}</p>
  </details>;
}

export default function Delay() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const action = useApiAction<DeliveryDurationOutput>();
  const purchaseDate = form.fecha_compra ? new Date(`${form.fecha_compra}T00:00:00Z`) : null;
  const period = purchaseDate && Number.isFinite(purchaseDate.getTime()) ? {
    month: MESES[purchaseDate.getUTCMonth()],
    weekday: DIAS[(purchaseDate.getUTCDay() + 6) % 7],
  } : null;
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
  const num = (key: FormKey, label: string, min = 0, step = "any", helpId?: string) => <FormField id={`delay-${key}`} label={label}>
    <input id={`delay-${key}`} className="input" type="number" min={min} step={step} required value={form[key]}
      aria-describedby={helpId} onChange={(e) => set(key, e.target.value)} />
  </FormField>;
  const select = (key: FormKey, label: string, choices: { value: string | number; label: string }[], placeholder?: string, helpId?: string) =>
    <FormField id={`delay-${key}`} label={label}>
      <select id={`delay-${key}`} className="input" required value={form[key]} aria-describedby={helpId} onChange={(e) => set(key, e.target.value)}>
        {placeholder && <option value="" disabled>{placeholder}</option>}
        {choices.map(({ value, label: optionLabel }) => <option key={value} value={value}>{optionLabel}</option>)}
      </select>
    </FormField>;
  return <>
    <PageHeader title="Tiempo de entrega" eyebrow="Predicción logística" subtitle="Estima cuántos días transcurrirán desde la compra hasta la entrega de un pedido." />
    <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1.45fr)_minmax(0,1fr)]">
      <form onSubmit={submit} className="card min-w-0 space-y-3" aria-labelledby="delay-form-title" aria-busy={action.loading}>
        <h2 id="delay-form-title" className="section-title sr-only">Detalles de la estimación</h2>
        <fieldset className="min-w-0">
          <legend className="section-title mb-2">Datos del pedido</legend>
          <div className="grid items-end gap-x-3 gap-y-2 sm:grid-cols-2 xl:grid-cols-4">
            {num("precio", "Precio total (BRL)")}{num("flete", "Flete (BRL)")}
            {num("peso_g", "Peso total (g)", 0, "any", "delay-weight-help")}
            {num("n_items", "Cantidad de artículos", 1, "1")}
          </div>
          <FieldHelp id="delay-weight-help" label="¿No conoces el peso?">Introduce 0 si no conoces el peso total.</FieldHelp>
        </fieldset>
        <div className="border-t border-slate-200 pt-3">
          <fieldset className="min-w-0">
            <legend className="section-title mb-2">Ruta logística</legend>
            <div className="grid gap-3 sm:grid-cols-2">
              {select("estado_vendedor", "Estado de origen", ESTADOS.map((value) => ({ value, label: value })), undefined, "delay-origin-help")}
              {select("estado_cliente", "Estado de destino", ESTADOS.map((value) => ({ value, label: value })))}
            </div>
            <FieldHelp id="delay-origin-help" label="¿Hay varios vendedores?">Usa el estado con mayor valor total de artículos. En empate, elige el de más artículos y luego el primero por sigla.</FieldHelp>
          </fieldset>
        </div>
        <div className="border-t border-slate-200 pt-3">
          <fieldset className="min-w-0">
            <legend className="section-title mb-2">Compra</legend>
            <div className={`grid items-end gap-x-3 gap-y-2 ${period ? "sm:grid-cols-2" : "sm:grid-cols-2 xl:grid-cols-3"}`}>
              <div className={period ? "min-w-0" : "min-w-0 sm:col-span-2 xl:col-span-1"}>
                <FormField id="delay-fecha_compra" label="Fecha de compra (opcional)">
                  <input id="delay-fecha_compra" className="input w-full min-w-0" type="date" value={form.fecha_compra}
                    aria-describedby={!form.fecha_compra ? "delay-fecha_compra-hint" : undefined} onChange={(e) => set("fecha_compra", e.target.value)} />
                </FormField>
              </div>
              {period && <p className="rounded-md bg-slate-50 px-3 py-2 text-sm text-slate-700" aria-live="polite">
                <span className="font-medium">Según la fecha:</span> {period.month} · {period.weekday}
              </p>}
              {!form.fecha_compra && <>
                {select("mes", "Mes de compra", MESES.map((label, i) => ({ value: i + 1, label })), "Selecciona el mes")}
                {select("dia_semana", "Día de la semana de compra", DIAS.map((label, i) => ({ value: i, label })), "Selecciona el día")}
              </>}
            </div>
            {!form.fecha_compra && <p id="delay-fecha_compra-hint" className="mt-2 text-xs leading-5 text-slate-600">Sin fecha exacta, completa mes y día para estimar solo la duración.</p>}
          </fieldset>
        </div>
        <button type="submit" className="btn w-full sm:w-auto" disabled={action.loading}>
          <Icon name="shield" className="h-4 w-4" />{action.loading ? "Estimando entrega…" : "Estimar tiempo de entrega"}
        </button>
      </form>
      <section className="card min-w-0" aria-labelledby="delay-result-title" aria-busy={action.loading}>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h2 id="delay-result-title" className="section-title">Resultado de tiempo de entrega</h2>
          {action.data && !action.loading && !action.error && <StatusBadge tone="info">Modelo entrenado</StatusBadge>}
        </div>
        <ErrorBox>{action.error}</ErrorBox>
        {action.loading ? <LoadingState label="Calculando la duración…" /> : action.data ? <div className="space-y-4" aria-live="polite">
          <dl className="space-y-3 rounded-lg bg-brand-50 p-4 sm:p-5">
            <div><dt className="text-sm text-slate-600">Tiempo estimado desde la compra</dt>
              <dd className="mt-1 text-5xl font-bold tracking-tight text-brand-700 tabular-nums">{number(action.data.dias_entrega_estimados, 2)} <span className="text-xl font-semibold tracking-normal">días</span></dd></div>
            <div className="border-t border-brand-100 pt-3"><dt className="text-sm text-slate-600">Fecha aproximada de llegada</dt>
              <dd className="font-semibold">{action.data.fecha_llegada_estimada
                ? <time dateTime={action.data.fecha_llegada_estimada}>{date(action.data.fecha_llegada_estimada)}</time>
                : "No disponible sin fecha de compra"}</dd></div>
          </dl>
          <p className="text-sm leading-5 text-slate-600">Evaluado con datos históricos. La estimación es demostrativa: no garantiza resultados comerciales ni representa una promesa de entrega o una probabilidad de retraso.</p>
          <div className="flex items-start gap-2 border-l-2 border-amber-500 pl-3 text-sm leading-5 text-slate-700">
            <Icon name="alert" className="mt-0.5 h-4 w-4 text-amber-700" />
            <p><strong>Menor fiabilidad en entregas prolongadas.</strong> El modelo puede subestimar su duración.</p>
          </div>
          <details className="border-t border-slate-200">
            <summary className="min-h-11 cursor-pointer py-3 text-sm font-semibold text-slate-700">Rendimiento del modelo</summary>
            <dl className="space-y-2 text-sm leading-5 text-slate-600">
              <div><dt className="font-medium text-slate-800">Algoritmo</dt><dd className="break-all">{action.data.modelo}</dd></div>
              <div><dt className="font-medium text-slate-800">MAE de desarrollo</dt><dd>{number(action.data.mae_desarrollo_dias, 2)} días</dd></div>
              <div><dt className="font-medium text-slate-800">MAE retrospectivo (agosto de 2018)</dt><dd>{number(action.data.mae_agosto_retrospectivo_dias, 2)} días</dd></div>
            </dl>
            <p className="mt-2 text-sm leading-5 text-slate-600">El error absoluto medio (MAE) resume resultados históricos; no es un intervalo de confianza para este pedido.</p>
          </details>
        </div> : !action.error && <EmptyState icon="shield" title="Listo para estimar" description="Completa las características del pedido para consultar su duración estimada." />}
      </section>
    </div>
  </>;
}
