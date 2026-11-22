// Cliente HTTP para la API de CommerceAI (FastAPI).
const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `Error ${res.status}`);
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
  health: () => request<Health>("/health"),
  predictDelay: (d: DelayInput) => request<DelayOutput>("/delay/predict", json(d)),
  forecast: (horizonte: number, historico = 90) =>
    request<DemandOutput>(`/demand/forecast?horizonte=${horizonte}&historico=${historico}`),
  classifyProduct: (file: File) => {
    const fd = new FormData();
    fd.append("imagen", file);
    return request<ProductOutput>("/products/classify", { method: "POST", body: fd });
  },
  sellers: () => request<SellersOutput>("/sellers/clusters"),
  summarize: (texto: string, n_oraciones: number) => request<SummaryOutput>("/reviews/summarize", json({ texto, n_oraciones })),
  summarizeSeller: (id: string, n: number) => request<SummaryOutput>(`/reviews/seller/${id}?n_oraciones=${n}`),
};
