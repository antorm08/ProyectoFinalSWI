import { useState } from "react";
import { api, type ProductOutput } from "../api/client";
import { ErrorBox, ModelBadge, PageHeader } from "../components/common";

export default function Products() {
  const [preview, setPreview] = useState<string>("");
  const [res, setRes] = useState<ProductOutput | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onFile = async (file?: File) => {
    if (!file) return;
    setPreview(URL.createObjectURL(file));
    setError("");
    setRes(null);
    setLoading(true);
    try { setRes(await api.classifyProduct(file)); } catch (e) { setError((e as Error).message); }
    setLoading(false);
  };

  return (
    <>
      <PageHeader title="Clasificación de productos" subtitle="Sube una foto de una prenda y la CNN propondrá su categoría" />
      <div className="grid gap-6 lg:grid-cols-2">
        <label className="card flex min-h-64 cursor-pointer flex-col items-center justify-center border-2 border-dashed text-slate-500 hover:border-indigo-400">
          {preview ? <img src={preview} alt="producto" className="max-h-64 object-contain" /> : "Haz clic para elegir una imagen"}
          <input type="file" accept="image/*" hidden onChange={(e) => onFile(e.target.files?.[0])} />
        </label>
        <div className="card">
          <h3 className="mb-3 font-semibold">Resultado</h3>
          <ErrorBox>{error}</ErrorBox>
          {loading && <p>Clasificando…</p>}
          {res && (
            <>
              <p className="text-3xl font-bold">{res.categoria}</p>
              {res.modelo_entrenado && <p className="text-slate-500">Confianza: {(res.confianza * 100).toFixed(1)}%</p>}
              <div className="my-4 space-y-2">
                {res.top3.map((t) => (
                  <div key={t.categoria}>
                    <div className="flex justify-between text-sm"><span>{t.categoria}</span><span>{(t.confianza * 100).toFixed(1)}%</span></div>
                    <div className="h-2 rounded bg-slate-100"><div className="h-2 rounded bg-indigo-500" style={{ width: `${t.confianza * 100}%` }} /></div>
                  </div>
                ))}
              </div>
              <ModelBadge trained={res.modelo_entrenado} />
            </>
          )}
        </div>
      </div>
    </>
  );
}
