import { useEffect, useState } from "react";
import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { api, type SellersOutput } from "../api/client";
import { ErrorBox, ModelBadge, PageHeader } from "../components/common";

const COLORS = ["#4f46e5", "#059669", "#d97706", "#dc2626", "#0891b2", "#7c3aed"];

export default function Sellers() {
  const [data, setData] = useState<SellersOutput | null>(null);
  const [error, setError] = useState("");

  useEffect(() => { api.sellers().then(setData).catch((e) => setError(e.message)); }, []);

  if (error) return <ErrorBox>{error}</ErrorBox>;
  if (!data) return <p>Cargando…</p>;

  return (
    <>
      <PageHeader title="Segmentación de vendedores" subtitle="Grupos de vendedores según volumen, satisfacción y puntualidad" />
      <ModelBadge trained={data.modelo_entrenado} />
      {!data.modelo_entrenado ? (
        <p className="mt-4 text-slate-500">Ejecuta <code>python ml/train_sellers.py</code> para generar los clusters.</p>
      ) : (
        <>
          <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {data.clusters.map((c) => (
              <div key={c.cluster} className="card" style={{ borderTop: `4px solid ${COLORS[c.cluster % COLORS.length]}` }}>
                <h3 className="font-semibold">{c.etiqueta}</h3>
                <p className="text-sm text-slate-500">{c.n_vendedores} vendedores</p>
                <ul className="mt-2 space-y-0.5 text-sm">
                  <li>Nota media: <b>{c.nota_media}</b></li>
                  <li>Tasa de retraso: <b>{(c.tasa_retraso * 100).toFixed(1)}%</b></li>
                  <li>Pedidos medios: <b>{c.n_pedidos}</b></li>
                  <li>Facturación media: <b>R$ {c.facturacion.toLocaleString()}</b></li>
                </ul>
              </div>
            ))}
          </div>
          <div className="card mt-6">
            <h3 className="mb-2 font-semibold">Nota media vs. tasa de retraso</h3>
            <ResponsiveContainer width="100%" height={380}>
              <ScatterChart>
                <CartesianGrid stroke="#e2e8f0" />
                <XAxis dataKey="tasa_retraso" name="Tasa de retraso" type="number" />
                <YAxis dataKey="nota_media" name="Nota media" type="number" domain={[1, 5]} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Legend />
                {data.clusters.map((c) => (
                  <Scatter key={c.cluster} name={c.etiqueta} fill={COLORS[c.cluster % COLORS.length]}
                    data={data.vendedores.filter((v) => v.cluster === c.cluster)} />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </>
  );
}
