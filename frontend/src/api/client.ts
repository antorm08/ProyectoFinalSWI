// Cliente HTTP para la API de CommerceAI (FastAPI).
const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch (error) {
    if (init?.signal?.aborted) throw error;
    throw new Error("No se pudo conectar con la API. Comprueba que el backend esté disponible e inténtalo de nuevo.");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail: unknown = body.detail;
    const message = typeof detail === "string" ? detail : Array.isArray(detail)
      ? detail.map((item) => `${Array.isArray(item.loc) ? item.loc.filter((part: unknown) => part !== "body" && part !== "query").join(" · ") : "Campo"}: ${item.msg ?? "Valor no válido"}`).join(". ")
      : `La API devolvió un error (${res.status}). Inténtalo de nuevo.`;
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

// ---- Tipos ----
export interface Health {
  status: string;
  datos_olist: Record<string, boolean>;
  modelos: Record<string, boolean>;
}
export interface DelayInput {
  precio: number; flete: number; peso_g: number; n_items: number; dias_estimados: number;
  estado_cliente: string; estado_vendedor: string; mes: number; dia_semana: number;
}
export interface DelayOutput { probabilidad_retraso: number; nivel_riesgo: string; modelo_entrenado: boolean }
export type DeliveryDurationInput = Pick<DelayInput, "precio" | "flete" | "peso_g" | "n_items" | "estado_cliente" | "estado_vendedor"> & (
  { fecha_compra: string; mes?: never; dia_semana?: never } |
  { fecha_compra?: never; mes: number; dia_semana: number }
);
export interface DeliveryDurationOutput {
  dias_entrega_estimados: number;
  fecha_llegada_estimada: string | null;
  modelo: string;
  experimental: boolean;
  mae_desarrollo_dias: number;
  mae_agosto_retrospectivo_dias: number;
}
export interface ForecastPoint { fecha: string; valor: number; tipo: "historico" | "pronostico" }
export interface DemandOutput { serie: ForecastPoint[]; modelo_entrenado: boolean }
export interface ProductOutput {
  categoria: string; confianza: number; top3: { categoria: string; confianza: number }[]; modelo_entrenado: boolean;
}
export interface SellerRow {
  seller_id?: string; cluster: number; etiqueta: string; n_pedidos: number; facturacion: number;
  nota_media: number; tasa_retraso: number; dias_entrega_medio: number; n_vendedores?: number;
}
export interface SellersOutput { modelo_entrenado: boolean; clusters: SellerRow[]; vendedores: SellerRow[] }
export interface SummaryOutput { resumen: string; oraciones_originales: number; oraciones_resumen: number }

// ---- Endpoints ----
export const api = {
  health: (signal?: AbortSignal) => request<Health>("/health", { signal }),
  predictDelay: (d: DelayInput, signal?: AbortSignal) => request<DelayOutput>("/delay/predict", { ...json(d), signal }),
  predictDeliveryDuration: (d: DeliveryDurationInput, signal?: AbortSignal) =>
    request<DeliveryDurationOutput>("/delay/predict-duration", { ...json(d), signal }),
  forecast: (horizonte: number, historico = 90, signal?: AbortSignal) =>
    request<DemandOutput>(`/demand/forecast?horizonte=${horizonte}&historico=${historico}`, { signal }),
  classifyProduct: (file: File, signal?: AbortSignal) => {
    const fd = new FormData();
    fd.append("imagen", file);
    return request<ProductOutput>("/products/classify", { method: "POST", body: fd, signal });
  },
  sellers: (signal?: AbortSignal) => request<SellersOutput>("/sellers/clusters", { signal }),
  summarize: (texto: string, n_oraciones: number, signal?: AbortSignal) => request<SummaryOutput>("/reviews/summarize", { ...json({ texto, n_oraciones }), signal }),
  summarizeSeller: (id: string, n: number, signal?: AbortSignal) => request<SummaryOutput>(`/reviews/seller/${encodeURIComponent(id)}?n_oraciones=${n}`, { signal }),
};
