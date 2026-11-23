import { useEffect, useRef, useState } from "react";
import { api, type ProductOutput } from "../api/client";
import { EmptyState, ErrorBox, FormField, LoadingState, Notice, PageHeader } from "../components/common";
import Icon from "../components/Icon";
import { useApiAction } from "../hooks/useApi";
import { number, percent } from "../utils/format";

// Fotos reales de producto (color, fondo blanco) que no forman parte de Fashion-MNIST.
const EJEMPLOS = [
  { slug: "camisa", label: "Camisa" },
  { slug: "pantalon", label: "Pantalón" },
  { slug: "vestido", label: "Vestido" },
  { slug: "zapatilla", label: "Zapatilla" },
  { slug: "bolso", label: "Bolso" },
  { slug: "botin", label: "Botín" },
] as const;
const ejemploUrl = (slug: string) => `${import.meta.env.BASE_URL}ejemplos/${slug}.jpg`;

export default function Products() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [validation, setValidation] = useState("");
  const [loadingExample, setLoadingExample] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const action = useApiAction<ProductOutput>();
  useEffect(() => {
    if (!file) { setPreview(""); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  const selectFile = (selected?: File) => {
    if (!selected) return;
    action.reset(); setValidation("");
    if (!selected.type.startsWith("image/")) {
      setFile(null); setValidation("Selecciona un archivo de imagen válido."); return;
    }
    setFile(selected);
    void action.run((signal) => api.classifyProduct(selected, signal));
  };
  const selectExample = async (slug: string) => {
    setLoadingExample(slug);
    try {
      const res = await fetch(ejemploUrl(slug));
      if (!res.ok) throw new Error();
      const blob = await res.blob();
      if (inputRef.current) inputRef.current.value = "";
      selectFile(new File([blob], `ejemplo-${slug}.jpg`, { type: blob.type || "image/jpeg" }));
    } catch {
      action.reset(); setFile(null); setValidation("No se pudo cargar la imagen de ejemplo. Inténtalo de nuevo.");
    } finally {
      setLoadingExample("");
    }
  };
  const result = action.data;
  return <>
    <PageHeader title="Clasificación visual" eyebrow="Catálogo" subtitle="Identifica la categoría de una prenda o accesorio." />
    <div className="mb-4"><Notice title="Solo prendas y accesorios">Usa una foto de una sola prenda sobre un fondo liso; la imagen se recorta automáticamente a la prenda. Otras categorías o fondos complejos pueden producir errores, incluso con confianza alta.</Notice></div>
    <div className="grid items-start gap-4 lg:grid-cols-2">
      <section className="card"><div className="mb-3 flex items-center gap-2"><span className="rounded-lg bg-brand-50 p-1.5 text-brand-600"><Icon name="upload" className="h-4 w-4" /></span><h2 className="section-title">Imagen</h2></div>
        <FormField id="product-image" label="Seleccionar imagen" hint="JPG, PNG o WebP, hasta 5 MB.">
          <input ref={inputRef} id="product-image" type="file" accept="image/*" className="input file:mr-3 file:rounded-md file:border-0 file:bg-brand-50 file:px-3 file:py-2 file:text-xs file:font-semibold file:text-brand-700" onChange={(e) => selectFile(e.target.files?.[0])} />
        </FormField>
        <fieldset className="mt-3">
          <legend className="mb-2 text-xs font-semibold text-slate-600">O prueba con un ejemplo</legend>
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
            {EJEMPLOS.map((ej) => <button key={ej.slug} type="button" onClick={() => void selectExample(ej.slug)} disabled={action.loading || loadingExample !== ""}
              aria-label={`Probar con el ejemplo: ${ej.label}`}
              className="flex min-h-11 flex-col items-center gap-1 rounded-lg border border-slate-200 bg-white p-1.5 text-xs text-slate-600 transition hover:border-brand-300 hover:bg-brand-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-600 disabled:cursor-wait disabled:opacity-60">
              <img src={ejemploUrl(ej.slug)} alt="" className="h-12 w-12 rounded object-contain" loading="lazy" />
              <span>{ej.label}</span>
            </button>)}
          </div>
        </fieldset>
        <div className="mt-3 overflow-hidden rounded-lg border border-dashed border-slate-300 bg-slate-50">
          {preview ? <figure><div className="flex min-h-44 items-center justify-center p-3"><img src={preview} alt={`Imagen seleccionada: ${file?.name ?? "prenda o accesorio"}`} className="max-h-56 max-w-full rounded-lg object-contain" /></div><figcaption className="border-t border-slate-200 bg-white px-3 py-2 text-xs text-slate-500"><p className="break-all font-medium text-slate-700">{file?.name}</p>{file && <p className="mt-1">{number(file.size / 1024, 1)} KB</p>}</figcaption></figure>
            : <EmptyState icon="image" title="Selecciona una imagen" description="Un solo objeto y un fondo sencillo ofrecen mejores condiciones de entrada." />}
        </div>
      </section>
      <section className="card" aria-label="Resultado de clasificación" aria-busy={action.loading}>
        <h2 className="section-title mb-3">Resultado</h2>
        <ErrorBox>{validation || action.error}</ErrorBox>
        {action.loading ? <LoadingState label="Clasificando imagen…" /> : result ? <div aria-live="polite">
          {result.modelo_entrenado ? <>
            {result.baja_confianza && <div className="mt-3"><Notice title="Confianza baja: revisa la categoría" tone="warning">La confianza del modelo ({percent(result.confianza)}) no alcanza el umbral validado de {percent(result.umbral_confianza)}. Comprueba manualmente la categoría o prueba con otra foto.</Notice></div>}
            <div className={`my-4 rounded-lg p-3 ${result.baja_confianza ? "bg-amber-50" : "bg-brand-50"}`}><p className={`text-xs font-semibold ${result.baja_confianza ? "text-amber-800" : "text-brand-700"}`}>Categoría propuesta</p><p className="mt-1 text-2xl font-bold tracking-tight">{result.categoria}</p><p className="mt-2 text-sm text-slate-600">Confianza del modelo: <strong className={result.baja_confianza ? "text-amber-800" : "text-brand-700"}>{percent(result.confianza)}</strong></p></div>
            <h3 className="mb-3 text-sm font-semibold">Categorías más probables</h3><ol className="space-y-3">{result.top3.map((item, i) => <li key={item.categoria}><div className="mb-1.5 flex items-center justify-between gap-3 text-sm"><span className="text-slate-700"><span className="mr-2 text-xs text-slate-500">0{i + 1}</span>{item.categoria}</span><strong className="tabular-nums">{percent(item.confianza)}</strong></div><div className="h-2 overflow-hidden rounded-full bg-slate-100" aria-hidden="true"><div className={`h-full rounded-full ${i === 0 ? (result.baja_confianza ? "bg-amber-500" : "bg-brand-600") : "bg-slate-400"}`} style={{ width: `${Math.max(0, Math.min(100, item.confianza * 100))}%` }} /></div></li>)}</ol>
            <p className="mt-4 border-t border-slate-100 pt-3 text-xs leading-5 text-slate-500">La confianza no equivale a precisión validada.</p>
          </> : <div className="mt-3"><Notice title="Clasificación no disponible" tone="warning">El modelo está pendiente. No se ha clasificado la imagen.</Notice></div>}
        </div> : !action.error && !validation && <EmptyState icon="layers" title="Sin resultado" description="La clasificación comienza al seleccionar una imagen o un ejemplo." />}
      </section>
    </div>
    <details className="card mt-4"><summary className="min-h-11 py-2.5 text-sm font-semibold text-slate-700">Acerca del clasificador</summary>
      <div className="space-y-2 text-sm leading-5 text-slate-500">
        <p>Red neuronal convolucional (CNN) entrenada con Fashion-MNIST: diez categorías de prendas y accesorios (camiseta/top, pantalón, jersey, vestido, abrigo, sandalia, camisa, zapatilla, bolso y botín) en imágenes de 28 × 28 píxeles en gris.</p>
        <p>Acierta el 92,5 % de las 10 000 imágenes de prueba. La categoría más difícil es camisa, que se confunde con camiseta, jersey y abrigo.</p>
        <p>Antes de clasificar, la foto se pasa a gris, se recorta a la prenda y se escala a 28 × 28. Si la confianza es inferior al umbral validado (70 %), el resultado se marca para revisión.</p>
      </div>
    </details>
  </>;
}
