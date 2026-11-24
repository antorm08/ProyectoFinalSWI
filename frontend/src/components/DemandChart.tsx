import { Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DemandMetrics, DemandOutput } from "../api/client";
import { useSystem } from "./SystemContext";
import { Notice } from "./common";
import { date, number } from "../utils/format";

const porcentaje = (nivel: number | null) => `${Math.round((nivel ?? 0.8) * 100)} %`;

export function DemandSource({ trained, metricas, modelo }: { trained: boolean; metricas?: DemandMetrics | null; modelo?: string }) {
  const { data, error, loading } = useSystem();
  if (trained) return <div className="space-y-1 text-xs leading-5 text-slate-600">
    {modelo && <p className="font-semibold">{modelo}</p>}
    <p>{metricas
      ? `Modelo medido fuera de muestra: ${number(metricas.mae, 1)} pedidos/día de error absoluto medio (${metricas.periodo}).`
      : "Estimación del modelo sobre su histórico de referencia; precisión no verificada."}</p>
    {metricas?.cobertura_80 != null && <p>{`Cobertura observada en evaluación: ${number(metricas.cobertura_80 * 100, 1)} % de los días para la banda nominal del 80 %.`}</p>}
  </div>;
  if (loading || error || !data) return <Notice title="Procedencia no confirmada" tone="warning">Pronóstico provisional. No se pudo distinguir entre histórico Olist y datos sintéticos; no asumir actividad real.</Notice>;
  return data.datos_olist.orders
    ? <Notice title="Estimación orientativa" tone="warning">Datos históricos de Olist. Sin modelo entrenado, se repite la última semana.</Notice>
    : <Notice title="Datos sintéticos · solo demostración" tone="warning">Histórico sintético y repetición de su última semana. No representa pedidos reales.</Notice>;
}

type Fila = {
  fecha: string; historico: number | null; pronostico: number | null;
  bandaBase: number | null; bandaRango: number | null;
  inferior: number | null; superior: number | null;
};

function Etiqueta({ active, payload, label, nivel }: { active?: boolean; payload?: { payload: Fila }[]; label?: string; nivel: number | null }) {
  if (!active || !payload?.length) return null;
  const fila = payload[0].payload;
  const valor = fila.historico ?? fila.pronostico;
  return <div className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-[13px] shadow-sm">
    <p className="font-semibold text-slate-700">{date(String(label))}</p>
    <p className="text-slate-600">{fila.historico !== null ? "Histórico" : "Pronóstico"}: {number(valor ?? 0, 2)} pedidos</p>
    {fila.inferior !== null && fila.superior !== null &&
      <p className="text-slate-500">Banda {porcentaje(nivel)}: {number(fila.inferior, 0)} – {number(fila.superior, 0)}</p>}
  </div>;
}

export default function DemandChart({ data, compact = false }: { data: DemandOutput; compact?: boolean }) {
  const nivel = data.nivel_confianza;
  const rows: Fila[] = data.serie.map((point) => {
    const esHistorico = point.tipo === "historico";
    const tieneBanda = !esHistorico && point.inferior !== null && point.superior !== null;
    return {
      fecha: point.fecha,
      historico: esHistorico ? point.valor : null,
      pronostico: esHistorico ? null : point.valor,
      // La banda se dibuja como dos áreas apiladas: una base invisible hasta el
      // extremo inferior y encima el ancho del intervalo.
      bandaBase: tieneBanda ? point.inferior : null,
      bandaRango: tieneBanda ? point.superior! - point.inferior! : null,
      inferior: tieneBanda ? point.inferior : null,
      superior: tieneBanda ? point.superior : null,
    };
  });
  const history = data.serie.filter((p) => p.tipo === "historico");
  const future = data.serie.filter((p) => p.tipo === "pronostico");
  const conBanda = future.some((p) => p.inferior !== null);
  return <div>
    <p className="mb-3 text-xs text-slate-500">{data.serie.length ? `${date(data.serie[0].fecha)} — ${date(data.serie[data.serie.length - 1].fecha)}` : "Sin período disponible"}</p>
    <div className={compact ? "h-48 w-full min-w-0 sm:h-52" : "h-60 w-full min-w-0 sm:h-64"} role="img" aria-label={`Serie de demanda: ${history.length} días históricos y ${future.length} días de pronóstico${conBanda ? `, con banda de incertidumbre del ${porcentaje(nivel)}` : ""}. Los valores también están disponibles en la tabla siguiente.`}>
      <ResponsiveContainer width="100%" height="100%" minWidth={0}>
        <ComposedChart data={rows} margin={{ top: 10, right: 12, left: -12, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#e2e8f0" />
          <XAxis dataKey="fecha" minTickGap={35} tick={{ fontSize: 11, fill: "#64748b" }} tickLine={false} axisLine={false} tickFormatter={(v: string) => date(v).replace(/ de \d{4}$/, "")} />
          <YAxis tick={{ fontSize: 11, fill: "#64748b" }} tickLine={false} axisLine={false} />
          <Tooltip content={<Etiqueta nivel={nivel} />} />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
          {conBanda && <Area dataKey="bandaBase" stackId="banda" stroke="none" fill="none" legendType="none" isAnimationActive={false} activeDot={false} tooltipType="none" />}
          {conBanda && <Area dataKey="bandaRango" stackId="banda" name={`Banda ${porcentaje(nivel)}`} stroke="none" fill="#2563eb" fillOpacity={0.14} isAnimationActive={false} activeDot={false} tooltipType="none" />}
          <Line dataKey="historico" name="Histórico" stroke="#64748b" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line dataKey="pronostico" name="Pronóstico" stroke="#2563eb" strokeWidth={2.5} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
    <details className="mt-2 border-t border-slate-100"><summary className="min-h-11 py-3 text-xs font-semibold text-brand-700">Ver datos del gráfico en tabla</summary>
      <div className="max-h-64 overflow-auto rounded-lg border border-slate-200" tabIndex={0} role="region" aria-label="Datos de la serie de demanda"><table className="data-table"><caption className="sr-only">Serie de demanda en pedidos por día; conserva el mismo carácter real o provisional del gráfico</caption><thead><tr><th scope="col">Fecha</th><th scope="col">Tipo</th><th scope="col" className="text-right">Pedidos / día</th>{conBanda && <><th scope="col" className="text-right">Mínimo previsto</th><th scope="col" className="text-right">Máximo previsto</th></>}</tr></thead><tbody>{data.serie.map((p) => <tr key={`${p.fecha}-${p.tipo}`}><td>{date(p.fecha)}</td><td>{p.tipo === "historico" ? "Histórico" : "Pronóstico"}</td><td className="text-right tabular-nums">{number(p.valor, 2)}</td>{conBanda && <><td className="text-right tabular-nums">{p.inferior === null ? "—" : number(p.inferior, 0)}</td><td className="text-right tabular-nums">{p.superior === null ? "—" : number(p.superior, 0)}</td></>}</tr>)}</tbody></table></div>
    </details>
  </div>;
}
