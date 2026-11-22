import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type DemandOutput } from "../api/client";
import { ErrorBox, ModelBadge, PageHeader } from "../components/common";

export default function Demand() {
  const [horizonte, setHorizonte] = useState(30);
  const [data, setData] = useState<DemandOutput | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.forecast(horizonte).then(setData).catch((e) => setError(e.message));
  }, [horizonte]);

  // Separar en dos series para pintar histórico y pronóstico con estilos distintos
  const rows = useMemo(
    () => data?.serie.map((p) => ({
      fecha: p.fecha,
      historico: p.tipo === "historico" ? p.valor : null,
      pronostico: p.tipo === "pronostico" ? p.valor : null,
    })) ?? [],
    [data],
  );
  const totalPronostico = data?.serie.filter((p) => p.tipo === "pronostico").reduce((s, p) => s + p.valor, 0) ?? 0;

  return (
    <>
      <PageHeader title="Demanda de pedidos" subtitle="Pronóstico de pedidos diarios a partir del histórico de Olist" />
      <ErrorBox>{error}</ErrorBox>
      <div className="card">
        <div className="mb-4 flex flex-wrap items-center gap-4">
          <label className="text-sm">Horizonte (días):</label>
          <input type="range" min={7} max={90} value={horizonte} onChange={(e) => setHorizonte(Number(e.target.value))} />
          <span className="font-mono">{horizonte}</span>
          {data && <ModelBadge trained={data.modelo_entrenado} />}
          <span className="ml-auto text-sm text-slate-500">
            Pedidos esperados: <b className="text-slate-800">{Math.round(totalPronostico).toLocaleString()}</b>
          </span>
        </div>
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="fecha" minTickGap={40} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line dataKey="historico" name="Histórico" stroke="#64748b" dot={false} />
            <Line dataKey="pronostico" name="Pronóstico" stroke="#4f46e5" strokeWidth={2} strokeDasharray="6 3" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
