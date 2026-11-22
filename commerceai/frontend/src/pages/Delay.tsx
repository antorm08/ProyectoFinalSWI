import { useState, type FormEvent } from "react";
import { api, type DelayInput, type DelayOutput } from "../api/client";
import { ErrorBox, ModelBadge, PageHeader } from "../components/common";

const ESTADOS = ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT","PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"];
const DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"];

const colores: Record<string, string> = { bajo: "text-emerald-600", medio: "text-amber-600", alto: "text-red-600" };

export default function Delay() {
  const [form, setForm] = useState<DelayInput>({
    precio: 120, flete: 20, peso_g: 1500, n_items: 1, dias_estimados: 20,
    estado_cliente: "RJ", estado_vendedor: "SP", mes: 11, dia_semana: 0,
  });
  const [res, setRes] = useState<DelayOutput | null>(null);
  const [error, setError] = useState("");

  const set = (k: keyof DelayInput, v: string) =>
    setForm({ ...form, [k]: k.startsWith("estado") ? v : Number(v) });

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    try { setRes(await api.predictDelay(form)); } catch (err) { setError((err as Error).message); }
  };

  const num = (k: keyof DelayInput, label: string) => (
    <div>
      <label className="label">{label}</label>
      <input className="input" type="number" value={form[k]} onChange={(e) => set(k, e.target.value)} />
    </div>
  );
  const select = (k: keyof DelayInput, label: string, opts: { v: string | number; t: string }[]) => (
    <div>
      <label className="label">{label}</label>
      <select className="input" value={form[k]} onChange={(e) => set(k, e.target.value)}>
        {opts.map((o) => <option key={o.v} value={o.v}>{o.t}</option>)}
      </select>
    </div>
  );

  return (
    <>
      <PageHeader title="Riesgo de retraso" subtitle="Estima la probabilidad de que un pedido llegue después de la fecha prometida" />
      <div className="grid gap-6 lg:grid-cols-3">
        <form onSubmit={submit} className="card grid gap-4 sm:grid-cols-2 lg:col-span-2">
          {num("precio", "Precio total (R$)")}
          {num("flete", "Flete (R$)")}
          {num("peso_g", "Peso (g)")}
          {num("n_items", "Nº de ítems")}
          {num("dias_estimados", "Días prometidos de entrega")}
          {select("mes", "Mes de compra", Array.from({ length: 12 }, (_, i) => ({ v: i + 1, t: String(i + 1) })))}
          {select("estado_cliente", "Estado del cliente", ESTADOS.map((e) => ({ v: e, t: e })))}
          {select("estado_vendedor", "Estado del vendedor", ESTADOS.map((e) => ({ v: e, t: e })))}
          {select("dia_semana", "Día de la compra", DIAS.map((d, i) => ({ v: i, t: d })))}
          <div className="flex items-end"><button className="btn w-full">Predecir</button></div>
        </form>
        <div className="card">
          <h3 className="mb-3 font-semibold">Resultado</h3>
          <ErrorBox>{error}</ErrorBox>
          {res ? (
            <>
              <p className={`text-5xl font-bold ${colores[res.nivel_riesgo]}`}>
                {(res.probabilidad_retraso * 100).toFixed(1)}%
              </p>
              <p className="mt-1 mb-3">Riesgo <b>{res.nivel_riesgo}</b></p>
              <ModelBadge trained={res.modelo_entrenado} />
            </>
          ) : <p className="text-slate-500">Completa el formulario y pulsa «Predecir».</p>}
        </div>
      </div>
    </>
  );
}
