import { useCallback, useState } from "react";
import { api } from "../api/client";
import { ChartCard, EmptyState, ErrorBox, FormField, LoadingState, MetricCard, PageHeader } from "../components/common";
import DemandChart, { DemandSource } from "../components/DemandChart";
import { useApiQuery } from "../hooks/useApi";
import { date, number } from "../utils/format";

export default function Demand() {
  const [horizon, setHorizon] = useState(30);
  const [selected, setSelected] = useState(30);
  const load = useCallback((signal: AbortSignal) => api.forecast(horizon, 90, signal), [horizon]);
  const query = useApiQuery(load);
  const future = query.data?.serie.filter((point) => point.tipo === "pronostico") ?? [];
  const total = future.reduce((sum, point) => sum + point.valor, 0);
  const pending = selected !== horizon;
  return <>
    <PageHeader title="Pronóstico de demanda" eyebrow="Series temporales" subtitle="Consulta la demanda diaria y ajusta el horizonte de pronóstico." />
    <form className="card mb-4 flex flex-wrap items-end gap-3" onSubmit={(e) => { e.preventDefault(); if (selected === horizon) query.retry(); else setHorizon(selected); }}>
      <div className="min-w-0 flex-1 basis-64"><FormField id="demand-horizon" label={`Horizonte de pronóstico: ${selected} días`} hint="De 7 a 90 días. Aplica los cambios para actualizar.">
        <input id="demand-horizon" type="range" min={7} max={90} value={selected} onChange={(e) => setSelected(Number(e.target.value))} className="mt-2 min-h-8 w-full" aria-describedby="demand-horizon-hint" />
      </FormField></div><button className="btn" type="submit" disabled={query.loading && !pending}>{query.loading && !pending ? "Consultando…" : "Aplicar horizonte"}</button>
    </form>
    {pending && <p className="mb-4 text-sm text-slate-600" role="status">Cambio pendiente: aplica {selected} días. Los resultados actuales corresponden a {horizon} días.</p>}
    <ErrorBox onRetry={query.retry}>{query.error}</ErrorBox>
    {query.loading && <div className="card"><LoadingState label={`Consultando pronóstico de ${horizon} días…`} /></div>}
    {query.data && <>
      <DemandSource trained={query.data.modelo_entrenado} />
      <div className="my-4 grid gap-3 md:grid-cols-3">
        <MetricCard label="Pedidos previstos" value={future.length ? number(total) : "—"} detail={`${future.length} días de pronóstico.`} />
        <MetricCard label="Promedio diario previsto" value={future.length ? number(total / future.length, 1) : "—"} detail="Pedidos/día, calculados con la serie." />
        <MetricCard label="Fin del horizonte" value={future.length ? date(future[future.length - 1].fecha) : "—"} detail="Fecha final de la serie." icon="layers" />
      </div>
      <ChartCard title="Evolución diaria de pedidos" description="Histórico continuo · pronóstico discontinuo · pedidos/día.">
        {query.data.serie.length ? <DemandChart data={query.data} /> : <EmptyState icon="chart" title="Serie no disponible" description="La API no devolvió puntos para este período." />}
      </ChartCard>
      <details className="card mt-4"><summary className="min-h-11 py-2.5 text-sm font-semibold text-slate-700">Acerca del pronóstico</summary><p className="text-sm leading-5 text-slate-500">{query.data.modelo_entrenado ? "Modelo SARIMA entrenado sobre su histórico de referencia." : "Modelo SARIMA pendiente; se repiten los valores de la última semana disponible."}</p></details>
    </>}
  </>;
}
