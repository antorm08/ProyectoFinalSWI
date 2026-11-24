import { Link } from "react-router-dom";
import { api } from "../api/client";
import { ChartCard, EmptyState, ErrorBox, LoadingState, MetricCard, PageHeader, StatusBadge } from "../components/common";
import DemandChart, { DemandSource } from "../components/DemandChart";
import Icon, { type IconName } from "../components/Icon";
import { useSystem } from "../components/SystemContext";
import { useApiQuery } from "../hooks/useApi";
import { date, number } from "../utils/format";

const modules: { key: string; to: string; title: string; technique: string; description: string; icon: IconName }[] = [
  { key: "tiempo_entrega", to: "/retrasos", title: "Tiempo de entrega", technique: "HistGradientBoosting · experimental", description: "Estima los días desde la compra hasta la entrega de un pedido.", icon: "shield" },
  { key: "demanda", to: "/demanda", title: "Pronóstico de demanda", technique: "SARIMA", description: "Consulta el volumen diario previsto.", icon: "chart" },
  { key: "productos", to: "/productos", title: "Clasificación visual", technique: "CNN · Fashion-MNIST", description: "Clasifica imágenes de prendas y accesorios.", icon: "image" },
  { key: "vendedores", to: "/vendedores", title: "Segmentación de vendedores", technique: "K-Means", description: "Compara desempeño comercial y logístico.", icon: "users" },
  { key: "resumen", to: "/resenas", title: "Resumen de reseñas", technique: "TextRank extractivo", description: "Resume opiniones de clientes.", icon: "text" },
];
const datasets: Record<string, string> = { orders: "Pedidos", items: "Detalle de pedidos", reviews: "Reseñas", products: "Productos", sellers: "Vendedores", customers: "Clientes" };
const loadForecast = (signal: AbortSignal) => api.forecast(14, 30, 0.8, signal);

export default function Dashboard() {
  const system = useSystem();
  const forecast = useApiQuery(loadForecast);
  const future = forecast.data?.serie.filter((point) => point.tipo === "pronostico") ?? [];
  const total = future.reduce((sum, point) => sum + point.valor, 0);
  const models = system.data ? ["tiempo_entrega", "demanda", "productos", "vendedores"].filter((key) => system.data!.modelos[key]).length : null;
  const available = system.data ? Object.values(system.data.datos_olist).filter(Boolean).length : null;
  return <>
    <PageHeader title="Dashboard ejecutivo" eyebrow="Visión general" subtitle="Demanda prevista y herramientas para tus operaciones." actions={<Link className="btn-secondary" to="/demanda">Explorar demanda<Icon name="arrow" className="h-4 w-4" /></Link>} />
    <ErrorBox onRetry={system.retry}>{system.error}</ErrorBox>
    <ChartCard title="Demanda prevista" description="30 días de histórico y 14 de pronóstico.">
      <ErrorBox onRetry={forecast.retry}>{forecast.error}</ErrorBox>
      {forecast.loading && <LoadingState label="Consultando demanda…" />}
      {forecast.data && <>
        <DemandSource trained={forecast.data.modelo_entrenado} metricas={forecast.data.metricas} modelo={forecast.data.modelo} />
        <div className="my-3 grid gap-3 md:grid-cols-3">
          <MetricCard label="Pedidos previstos" value={future.length ? number(total) : "—"} detail={`${future.length} días de pronóstico.`} />
          <MetricCard label="Promedio diario" value={future.length ? number(total / future.length, 1) : "—"} detail="Pedidos por día." />
          <MetricCard label="Fin del período" value={future.length ? date(future[future.length - 1].fecha) : "—"} detail="Último día del pronóstico." icon="layers" />
        </div>
        {forecast.data.serie.length ? <DemandChart data={forecast.data} compact /> : <EmptyState icon="chart" title="Sin datos de demanda" description="No hay valores disponibles para este período." />}
      </>}
    </ChartCard>
    <section className="mt-5"><h2 className="section-title mb-3">Herramientas de análisis</h2>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">{modules.map((module) => <Link key={module.key} to={module.to} className="card group transition-colors hover:border-blue-300"><div className="flex items-center gap-2"><span className="rounded-lg bg-brand-50 p-1.5 text-brand-600"><Icon name={module.icon} className="h-4 w-4" /></span><h3 className="min-w-0 flex-1 text-sm font-semibold">{module.title}</h3><Icon name="arrow" className="h-4 w-4 text-slate-400 group-hover:text-brand-600" /></div><p className="mt-2 text-sm leading-5 text-slate-500">{module.description}</p></Link>)}</div>
    </section>
    <details className="card mt-4">
      <summary className="section-title min-h-11 py-2.5">Estado del sistema</summary>
      {system.loading && <LoadingState label="Comprobando disponibilidad…" />}
      {!system.loading && !system.data && <p className="text-sm text-slate-600">Estado no confirmado</p>}
      {system.data && <div className="mt-3 space-y-4">
        <div className="grid gap-3 md:grid-cols-3">
          <MetricCard label="Modelos detectados" value={`${models} / 4`} detail="Archivos presentes; precisión no verificada." icon="layers" />
          <MetricCard label="Fuentes Olist" value={`${available} / 6`} detail="Disponibilidad de archivos, no de calidad." icon="database" />
          <MetricCard label="Resumen de texto" value={system.data.modelos.resumen ? "Disponible" : "No disponible"} detail="TextRank no requiere entrenamiento." icon="text" />
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          <section><h2 className="section-title mb-2">Datos de referencia</h2><ul className="grid grid-cols-2 gap-2">{Object.entries(datasets).map(([key, label]) => <li key={key} className="flex flex-wrap items-center justify-between gap-1.5 rounded-lg border border-slate-100 px-2.5 py-2"><span className="text-sm text-slate-700">{label}</span><StatusBadge tone={system.data!.datos_olist[key] ? "success" : "neutral"}>{system.data!.datos_olist[key] ? "Detectado" : "Pendiente"}</StatusBadge></li>)}</ul></section>
          <section><h2 className="section-title mb-2">Métodos y modelos</h2><ul className="divide-y divide-slate-100">{modules.map((module) => {
            const present = system.data!.modelos[module.key];
            return <li key={module.key} className="flex flex-wrap items-center justify-between gap-2 py-2"><div><p className="text-sm text-slate-700">{module.title}</p><p className="text-xs text-slate-500">{module.technique}</p></div><StatusBadge tone={present ? "success" : "warning"}>{module.key === "resumen" ? (present ? "Disponible" : "No disponible") : present ? "Modelo detectado" : "Modelo pendiente"}</StatusBadge></li>;
          })}</ul></section>
        </div>
      </div>}
    </details>
  </>;
}
