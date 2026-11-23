import { useCallback, useState } from "react";
import { api } from "../api/client";
import { ChartCard, EmptyState, ErrorBox, FormField, LoadingState, MetricCard, PageHeader } from "../components/common";
import DemandChart, { DemandSource } from "../components/DemandChart";
import { useApiQuery } from "../hooks/useApi";
import { date, number } from "../utils/format";

const NIVELES = [
  { valor: 0.8, etiqueta: "80 %" },
  { valor: 0.95, etiqueta: "95 %" },
];

export default function Demand() {
  const [horizon, setHorizon] = useState(30);
  const [selected, setSelected] = useState(30);
  const [nivel, setNivel] = useState(0.8);
  const load = useCallback((signal: AbortSignal) => api.forecast(horizon, 90, nivel, signal), [horizon, nivel]);
  const query = useApiQuery(load);
  const future = query.data?.serie.filter((point) => point.tipo === "pronostico") ?? [];
  const total = future.reduce((sum, point) => sum + point.valor, 0);
  const banda = future.filter((p) => p.inferior !== null && p.superior !== null);
  const totalMin = banda.reduce((sum, p) => sum + (p.inferior ?? 0), 0);
  const totalMax = banda.reduce((sum, p) => sum + (p.superior ?? 0), 0);
  const metricas = query.data?.metricas ?? null;
  const pending = selected !== horizon;
  return <>
    <PageHeader title="Pronóstico de demanda" eyebrow="Series temporales" subtitle="Consulta la demanda diaria y ajusta el horizonte de pronóstico." />
    <form className="card mb-4 flex flex-wrap items-end gap-3" onSubmit={(e) => { e.preventDefault(); if (selected === horizon) query.retry(); else setHorizon(selected); }}>
      <div className="min-w-0 flex-1 basis-64"><FormField id="demand-horizon" label={`Horizonte de pronóstico: ${selected} días`} hint="De 7 a 90 días. Aplica los cambios para actualizar.">
        <input id="demand-horizon" type="range" min={7} max={90} value={selected} onChange={(e) => setSelected(Number(e.target.value))} className="mt-2 min-h-8 w-full" aria-describedby="demand-horizon-hint" />
      </FormField></div>
      <div className="min-w-0 basis-48"><FormField id="demand-nivel" label="Nivel de la banda" hint="Cuánta incertidumbre cubre el intervalo.">
        <select id="demand-nivel" className="input mt-2" value={nivel} onChange={(e) => setNivel(Number(e.target.value))} aria-describedby="demand-nivel-hint">
          {NIVELES.map((n) => <option key={n.valor} value={n.valor}>{n.etiqueta}</option>)}
        </select>
      </FormField></div>
      <button className="btn" type="submit" disabled={query.loading && !pending}>{query.loading && !pending ? "Consultando…" : "Aplicar horizonte"}</button>
    </form>
    {pending && <p className="mb-4 text-sm text-slate-600" role="status">Cambio pendiente: aplica {selected} días. Los resultados actuales corresponden a {horizon} días.</p>}
    <ErrorBox onRetry={query.retry}>{query.error}</ErrorBox>
    {query.loading && <div className="card"><LoadingState label={`Consultando pronóstico de ${horizon} días…`} /></div>}
    {query.data && <>
      <DemandSource trained={query.data.modelo_entrenado} metricas={metricas} nivel={query.data.nivel_confianza} />
      {query.data.ancla && <p className="mt-2 text-xs leading-5 text-slate-500">El pronóstico arranca el día siguiente al último con datos ({date(query.data.ancla)}), no a partir de hoy: Olist es un archivo histórico cerrado.</p>}
      <div className="my-4 grid gap-3 md:grid-cols-3">
        <MetricCard label="Pedidos previstos" value={future.length ? number(total) : "—"} detail={banda.length ? `Entre ${number(totalMin)} y ${number(totalMax)} según la banda.` : `${future.length} días de pronóstico.`} />
        <MetricCard label="Promedio diario previsto" value={future.length ? number(total / future.length, 1) : "—"} detail="Pedidos/día, calculados con la serie." />
        <MetricCard label={metricas ? "Error típico del modelo" : "Fin del horizonte"} value={metricas ? `± ${number(metricas.mae, 1)}` : (future.length ? date(future[future.length - 1].fecha) : "—")} detail={metricas ? `Pedidos/día de error absoluto medio fuera de muestra. ${metricas.rel_mae < 1 ? "Mejor" : "Peor"} que repetir la última semana (×${number(metricas.rel_mae, 2)}).` : "Fecha final de la serie."} icon="layers" />
      </div>
      <ChartCard title="Evolución diaria de pedidos" description={`Histórico continuo · pronóstico discontinuo${banda.length ? " con su banda de incertidumbre" : ""} · pedidos/día.`}>
        {query.data.serie.length ? <DemandChart data={query.data} /> : <EmptyState icon="chart" title="Serie no disponible" description="La API no devolvió puntos para este período." />}
      </ChartCard>
      <details className="card mt-4"><summary className="min-h-11 py-2.5 text-sm font-semibold text-slate-700">Acerca del pronóstico</summary>
        <div className="space-y-2 text-sm leading-5 text-slate-500">
          <p>{query.data.modelo_entrenado
            ? `Modelo ${query.data.modelo}, entrenado sobre ${query.data.ventana?.inicio ? `el período ${date(query.data.ventana.inicio)} – ${date(query.data.ventana.fin ?? "")}` : "su histórico de referencia"}.`
            : "Modelo SARIMA pendiente; se repiten los valores de la última semana disponible."}</p>
          {metricas && <p>Medido en {metricas.periodo}, fuera de muestra: {number(metricas.mae, 1)} pedidos/día de error absoluto medio y MASE {number(metricas.mase, 2)}. El baseline de repetir la última semana comete {number(metricas.baseline_mae ?? 0, 1)}.</p>}
          {banda.length > 0 && <p>La banda no es decorativa: su cobertura se mide frente al nivel nominal. Análisis completo en <code>docs/evaluacion_demanda.md</code>.</p>}
        </div>
      </details>
    </>}
  </>;
}
