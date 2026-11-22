import { useState } from "react";
import { api, type SummaryOutput } from "../api/client";
import { ErrorBox, PageHeader } from "../components/common";

export default function Reviews() {
  const [modo, setModo] = useState<"texto" | "vendedor">("texto");
  const [texto, setTexto] = useState("");
  const [sellerId, setSellerId] = useState("");
  const [n, setN] = useState(3);
  const [res, setRes] = useState<SummaryOutput | null>(null);
  const [error, setError] = useState("");

  const run = async () => {
    setError("");
    setRes(null);
    try {
      setRes(modo === "texto" ? await api.summarize(texto, n) : await api.summarizeSeller(sellerId.trim(), n));
    } catch (e) { setError((e as Error).message); }
  };

  return (
    <>
      <PageHeader title="Resumen de reseñas" subtitle="Resume opiniones y reclamos de clientes (español o portugués)" />
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="card space-y-4">
          <div className="flex gap-2">
            {(["texto", "vendedor"] as const).map((m) => (
              <button key={m} onClick={() => setModo(m)}
                className={`rounded-lg px-3 py-1 text-sm ${modo === m ? "bg-indigo-600 text-white" : "bg-slate-100"}`}>
                {m === "texto" ? "Pegar texto" : "Reseñas de un vendedor"}
              </button>
            ))}
          </div>
          {modo === "texto" ? (
            <textarea className="input h-64" placeholder="Pega aquí las reseñas o un texto extenso…"
              value={texto} onChange={(e) => setTexto(e.target.value)} />
          ) : (
            <input className="input" placeholder="seller_id de Olist" value={sellerId} onChange={(e) => setSellerId(e.target.value)} />
          )}
          <div className="flex items-center gap-3">
            <label className="text-sm">Oraciones del resumen:</label>
            <input type="number" min={1} max={15} className="input w-20" value={n} onChange={(e) => setN(Number(e.target.value))} />
            <button className="btn ml-auto" onClick={run}>Resumir</button>
          </div>
        </div>
        <div className="card">
          <h3 className="mb-3 font-semibold">Resumen</h3>
          <ErrorBox>{error}</ErrorBox>
          {res && (
            <>
              <p className="leading-relaxed">{res.resumen}</p>
              <p className="mt-4 text-sm text-slate-500">
                {res.oraciones_resumen} de {res.oraciones_originales} oraciones
              </p>
            </>
          )}
        </div>
      </div>
    </>
  );
}
