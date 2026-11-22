import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DemandOutput } from "../api/client";
import { useSystem } from "./SystemContext";
import { Notice } from "./common";
import { date, number } from "../utils/format";

export function DemandSource({ trained }: { trained: boolean }) {
  const { data, error, loading } = useSystem();
  if (trained) return <p className="text-xs leading-5 text-slate-500">Estimación del modelo sobre su histórico de referencia; precisión no verificada.</p>;
  if (loading || error || !data) return <Notice title="Procedencia no confirmada" tone="warning">Pronóstico provisional. No se pudo distinguir entre histórico Olist y datos sintéticos; no asumir actividad real.</Notice>;
  return data.datos_olist.orders
    ? <Notice title="Estimación orientativa" tone="warning">Datos históricos de Olist. Sin modelo entrenado, se repite la última semana.</Notice>
    : <Notice title="Datos sintéticos · solo demostración" tone="warning">Histórico sintético y repetición de su última semana. No representa pedidos reales.</Notice>;
}

export default function DemandChart({ data, compact = false }: { data: DemandOutput; compact?: boolean }) {
  const rows = data.serie.map((point) => ({ fecha: point.fecha, historico: point.tipo === "historico" ? point.valor : null, pronostico: point.tipo === "pronostico" ? point.valor : null }));
  const history = data.serie.filter((p) => p.tipo === "historico");
  const future = data.serie.filter((p) => p.tipo === "pronostico");
  return <div>
    <p className="mb-3 text-xs text-slate-500">{data.serie.length ? `${date(data.serie[0].fecha)} — ${date(data.serie[data.serie.length - 1].fecha)}` : "Sin período disponible"}</p>
    <div className={compact ? "h-48 w-full min-w-0 sm:h-52" : "h-60 w-full min-w-0 sm:h-64"} role="img" aria-label={`Serie de demanda: ${history.length} días históricos y ${future.length} días de pronóstico. Los valores también están disponibles en la tabla siguiente.`}>
      <ResponsiveContainer width="100%" height="100%" minWidth={0}>
        <LineChart data={rows} margin={{ top: 10, right: 12, left: -12, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#e2e8f0" />
          <XAxis dataKey="fecha" minTickGap={35} tick={{ fontSize: 11, fill: "#64748b" }} tickLine={false} axisLine={false} tickFormatter={(v: string) => date(v).replace(/ de \d{4}$/, "")} />
          <YAxis tick={{ fontSize: 11, fill: "#64748b" }} tickLine={false} axisLine={false} />
          <Tooltip labelFormatter={(v) => date(String(v))} formatter={(value) => [number(Number(value), 2), undefined]} contentStyle={{ borderRadius: 12, borderColor: "#e2e8f0", fontSize: 13 }} />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
          <Line dataKey="historico" name="Histórico" stroke="#64748b" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line dataKey="pronostico" name="Pronóstico" stroke="#2563eb" strokeWidth={2.5} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
    <details className="mt-2 border-t border-slate-100"><summary className="min-h-11 py-3 text-xs font-semibold text-brand-700">Ver datos del gráfico en tabla</summary>
      <div className="max-h-64 overflow-auto rounded-lg border border-slate-200" tabIndex={0} role="region" aria-label="Datos de la serie de demanda"><table className="data-table"><caption className="sr-only">Serie de demanda en pedidos por día; conserva el mismo carácter real o provisional del gráfico</caption><thead><tr><th scope="col">Fecha</th><th scope="col">Tipo</th><th scope="col" className="text-right">Pedidos / día</th></tr></thead><tbody>{data.serie.map((p) => <tr key={`${p.fecha}-${p.tipo}`}><td>{date(p.fecha)}</td><td>{p.tipo === "historico" ? "Histórico" : "Pronóstico"}</td><td className="text-right tabular-nums">{number(p.valor, 2)}</td></tr>)}</tbody></table></div>
    </details>
  </div>;
}
