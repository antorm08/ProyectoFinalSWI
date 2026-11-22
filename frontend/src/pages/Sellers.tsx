import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import { ChartCard, EmptyState, ErrorBox, LoadingState, MetricCard, Notice, PageHeader, StatusBadge } from "../components/common";
import { useApiQuery } from "../hooks/useApi";
import { brl, number, percent } from "../utils/format";

const COLORS = ["#2563eb", "#047857", "#b45309", "#be123c", "#0e7490", "#7e22ce"];
const load = (signal: AbortSignal) => api.sellers(signal);

export default function Sellers() {
  const query = useApiQuery(load);
  const data = query.data;
  return <>
    <PageHeader title="Segmentación de vendedores" eyebrow="Inteligencia comercial" subtitle="Compara grupos por ventas, satisfacción y puntualidad." />
    <ErrorBox onRetry={query.retry}>{query.error}</ErrorBox>
    {query.loading && <div className="card"><LoadingState label="Consultando segmentos y vendedores…" /></div>}
    {data && <>
      {!data.modelo_entrenado ? <section className="card"><EmptyState icon="users" title="Segmentación no disponible" description="El análisis de vendedores aún no está preparado." /><details className="mt-2"><summary className="min-h-11 py-3 text-sm font-semibold text-slate-700">Detalles técnicos</summary><p className="break-words text-sm leading-5 text-slate-500">K-Means sobre datos Olist. El resumen requiere <code>seller_clusters.csv</code>; la asignación requiere <code>seller_kmeans.joblib</code>.</p></details></section> : <>
        <div className="mb-4"><Notice title="Datos históricos · BRL">Resúmenes sobre la tabla procesada. El gráfico y el detalle incluyen hasta 300 vendedores de mayor facturación, no toda la población.</Notice></div>
        <div className="mb-4 grid gap-3 md:grid-cols-3"><MetricCard label="Grupos de vendedores" value={number(data.clusters.length)} detail="Grupos del análisis disponible." icon="layers" /><MetricCard label="Vendedores analizados" value={number(data.clusters.reduce((sum, cluster) => sum + (cluster.n_vendedores ?? 0), 0))} detail="Total incluido en los grupos." icon="users" /><MetricCard label="Vendedores en esta vista" value={number(data.vendedores.length)} detail="Selección del gráfico y el detalle." icon="database" /></div>
        <div className="mb-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{data.clusters.map((cluster) => <section className="card relative overflow-hidden" key={cluster.cluster}><div className="absolute inset-x-0 top-0 h-1" style={{ background: COLORS[cluster.cluster % COLORS.length] }} /><p className="eyebrow">Segmento {cluster.cluster + 1}</p><h2 className="mt-1 text-sm font-semibold">{cluster.etiqueta}</h2><p className="mt-1 text-xs text-slate-500">{number(cluster.n_vendedores ?? 0)} vendedores</p><dl className="mt-3 space-y-2 text-sm"><div className="flex justify-between gap-3"><dt className="text-slate-500">Nota media</dt><dd className="font-semibold tabular-nums">{number(cluster.nota_media, 2)} / 5</dd></div><div className="flex justify-between gap-3"><dt className="text-slate-500">Retrasos</dt><dd className="font-semibold tabular-nums">{percent(cluster.tasa_retraso)}</dd></div><div className="flex justify-between gap-3"><dt className="text-slate-500">Pedidos medios</dt><dd className="font-semibold tabular-nums">{number(cluster.n_pedidos, 1)}</dd></div><div><dt className="text-slate-500">Facturación media</dt><dd className="mt-1 font-semibold tabular-nums">{brl(cluster.facturacion)}</dd></div></dl></section>)}</div>
        <ChartCard title="Satisfacción y puntualidad" description="Un punto por vendedor de la selección disponible.">
          {data.vendedores.length ? <div className="h-60 min-w-0 sm:h-72" role="img" aria-label="Dispersión de vendedores: tasa de retraso en el eje horizontal y nota media en el vertical. Consulta los mismos datos en la tabla inferior."><ResponsiveContainer width="100%" height="100%" minWidth={0}><ScatterChart margin={{ left: -12, right: 12, bottom: 16, top: 10 }}>
            <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 5" /><XAxis dataKey="tasa_retraso" name="Tasa de retraso" type="number" tickFormatter={percent} tick={{ fontSize: 11 }} /><YAxis dataKey="nota_media" name="Nota media" type="number" domain={[1, 5]} tick={{ fontSize: 11 }} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(value, name) => [name === "Tasa de retraso" ? percent(Number(value)) : number(Number(value), 2), name]} contentStyle={{ borderRadius: 12, fontSize: 12 }} /><Legend wrapperStyle={{ fontSize: 12, paddingTop: 16 }} />
            {data.clusters.map((cluster) => <Scatter key={cluster.cluster} name={`S${cluster.cluster + 1} · ${cluster.etiqueta}`} fill={COLORS[cluster.cluster % COLORS.length]} shape={["circle", "diamond", "square", "triangle"][cluster.cluster % 4] as "circle" | "diamond" | "square" | "triangle"} isAnimationActive={false} data={data.vendedores.filter((seller) => seller.cluster === cluster.cluster)} />)}
          </ScatterChart></ResponsiveContainer></div> : <EmptyState title="Sin vendedores para mostrar" description="No hay registros en la selección disponible." icon="users" />}
        </ChartCard>
        <section className="card mt-4"><div className="mb-3 flex flex-wrap items-center justify-between gap-3"><div><h2 className="section-title">Detalle de vendedores</h2><p className="mt-1 text-sm text-slate-500">Mayor facturación primero.</p></div><StatusBadge>{data.vendedores.length} registros</StatusBadge></div>
          <div className="max-h-120 overflow-auto rounded-lg border border-slate-200" tabIndex={0} role="region" aria-label="Tabla desplazable de vendedores"><table className="data-table"><caption className="sr-only">Vendedores del subconjunto de mayor facturación, no la población completa</caption><thead className="sticky top-0"><tr>{["Vendedor", "Segmento", "Pedidos", "Facturación (BRL)", "Nota / 5", "Retrasos", "Entrega (días)"].map((header, i) => <th key={header} scope="col" className={i > 1 ? "text-right" : ""}>{header}</th>)}</tr></thead><tbody>{data.vendedores.map((seller, i) => <tr key={seller.seller_id ?? `${seller.cluster}-${i}`}><td className="font-mono text-xs">{seller.seller_id ?? "No informado"}</td><td>S{seller.cluster + 1} · {seller.etiqueta}</td><td className="text-right tabular-nums">{number(seller.n_pedidos)}</td><td className="text-right tabular-nums">{brl(seller.facturacion)}</td><td className="text-right tabular-nums">{number(seller.nota_media, 2)}</td><td className="text-right tabular-nums">{percent(seller.tasa_retraso)}</td><td className="text-right tabular-nums">{number(seller.dias_entrega_medio, 1)}</td></tr>)}</tbody></table></div>
          {!data.vendedores.length && <EmptyState title="No hay filas disponibles" description="Los segmentos pueden estar disponibles aunque no se hayan devuelto vendedores." icon="users" />}
        </section>
      </>}
    </>}
  </>;
}
