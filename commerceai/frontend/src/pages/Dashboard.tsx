import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type Health } from "../api/client";
import { ErrorBox, PageHeader } from "../components/common";

const modulos = [
  { key: "retraso", to: "/retrasos", titulo: "Riesgo de retraso", tecnica: "Predicción · Gradient Boosting" },
  { key: "demanda", to: "/demanda", titulo: "Demanda de pedidos", tecnica: "Series temporales · SARIMA" },
  { key: "productos", to: "/productos", titulo: "Categorías por imagen", tecnica: "Clasificación · CNN" },
  { key: "vendedores", to: "/vendedores", titulo: "Segmentos de vendedores", tecnica: "Clustering · K-Means" },
  { key: "resumen", to: "/resenas", titulo: "Resumen de reseñas", tecnica: "Resumen de texto · TextRank" },
];

export default function Dashboard() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(`No se pudo conectar con la API: ${e.message}`));
  }, []);

  return (
    <>
      <PageHeader title="Panel general" subtitle="Plataforma inteligente para pedidos, entregas y satisfacción de clientes" />
      <ErrorBox>{error}</ErrorBox>
      <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {modulos.map((m) => {
          const ok = health?.modelos[m.key];
          return (
            <Link key={m.key} to={m.to} className="card transition hover:border-indigo-400">
              <div className="flex items-start justify-between">
                <h3 className="font-semibold">{m.titulo}</h3>
                <span className={`h-3 w-3 rounded-full ${ok ? "bg-emerald-500" : health ? "bg-amber-400" : "bg-slate-300"}`} />
              </div>
              <p className="mt-1 text-sm text-slate-500">{m.tecnica}</p>
            </Link>
          );
        })}
      </div>
      {health && (
        <div className="card mt-6">
          <h3 className="mb-2 font-semibold">Datos de Olist cargados</h3>
          <div className="flex flex-wrap gap-2 text-sm">
            {Object.entries(health.datos_olist).map(([k, v]) => (
              <span key={k} className={`rounded px-2 py-1 ${v ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"}`}>
                {k} {v ? "✓" : "✗"}
              </span>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
