import type { ReactNode } from "react";
import Icon, { type IconName } from "./Icon";

export function PageHeader({ title, subtitle, eyebrow = "Inteligencia operativa", actions }: { title: string; subtitle: string; eyebrow?: string; actions?: ReactNode }) {
  return (
    <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
      <div className="max-w-3xl"><p className="eyebrow mb-1">{eyebrow}</p>
        <h1 className="text-xl font-bold tracking-tight sm:text-2xl">{title}</h1>
        <p className="mt-1 text-sm leading-5 text-slate-600">{subtitle}</p>
      </div>
      {actions}
    </header>
  );
}

export function StatusBadge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "success" | "warning" | "error" | "info" }) {
  const colors = { neutral: "bg-slate-100 text-slate-700 ring-slate-200", success: "bg-emerald-50 text-emerald-800 ring-emerald-200", warning: "bg-amber-50 text-amber-900 ring-amber-200", error: "bg-red-50 text-red-800 ring-red-200", info: "bg-brand-50 text-brand-700 ring-brand-100" };
  return <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${colors[tone]}`}><Icon name={tone === "success" ? "check" : tone === "warning" || tone === "error" ? "alert" : "layers"} className="h-3.5 w-3.5" />{children}</span>;
}

export function ModelBadge({ trained }: { trained: boolean }) {
  return <StatusBadge tone={trained ? "success" : "warning"}>{trained ? "Modelo entrenado" : "Provisional · sin modelo entrenado"}</StatusBadge>;
}

export function ErrorBox({ children, onRetry }: { children: ReactNode; onRetry?: () => void }) {
  return children ? <div role="alert" className="mb-3 flex flex-wrap items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
    <Icon name="alert" className="h-4 w-4" /><div className="min-w-0 flex-1 break-words leading-5">{children}</div>
    {onRetry && <button type="button" className="min-h-11 rounded-lg px-3 font-semibold underline underline-offset-4" onClick={onRetry}>Reintentar</button>}
  </div> : null;
}

export function Notice({ title, children, tone = "info" }: { title: string; children: ReactNode; tone?: "info" | "warning" }) {
  return <div className={`flex gap-2 rounded-lg border p-3 text-sm ${tone === "warning" ? "border-amber-200 bg-amber-50 text-amber-950" : "border-brand-100 bg-brand-50 text-blue-950"}`}>
    <Icon name={tone === "warning" ? "alert" : "database"} className="h-4 w-4" /><div className="min-w-0"><p className="font-semibold">{title}</p><div className="mt-0.5 leading-5">{children}</div></div>
  </div>;
}

export function MetricCard({ label, value, detail, icon = "chart" }: { label: string; value: ReactNode; detail: string; icon?: IconName }) {
  return <div className="card"><div className="flex items-center justify-between gap-2"><p className="text-sm font-medium text-slate-600">{label}</p><span className="rounded-lg bg-brand-50 p-1.5 text-brand-600"><Icon name={icon} className="h-4 w-4" /></span></div>
    <p className="mt-2 break-words text-2xl font-bold tracking-tight tabular-nums">{value}</p><p className="mt-1 text-xs leading-5 text-slate-500">{detail}</p>
  </div>;
}

export function EmptyState({ title, description, icon = "layers" }: { title: string; description: string; icon?: IconName }) {
  return <div className="flex min-h-36 flex-col items-center justify-center px-3 py-4 text-center"><span className="mb-2 rounded-xl bg-slate-100 p-2 text-slate-500"><Icon name={icon} className="h-5 w-5" /></span><p className="text-sm font-semibold text-slate-800">{title}</p><p className="mt-1 max-w-md text-sm leading-5 text-slate-500">{description}</p></div>;
}

export function LoadingState({ label = "Consultando información…" }: { label?: string }) {
  return <div role="status" className="flex min-h-24 items-center justify-center gap-2 text-sm text-slate-600"><Icon name="refresh" className="h-4 w-4 animate-spin motion-reduce:animate-none" />{label}</div>;
}

export function FormField({ id, label, hint, children }: { id: string; label: string; hint?: string; children: ReactNode }) {
  return <div className="min-w-0"><label htmlFor={id} className="label">{label}</label>{children}{hint && <p id={`${id}-hint`} className="mt-1 text-xs leading-5 text-slate-500">{hint}</p>}</div>;
}

export function ChartCard({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return <section className="card"><h2 className="section-title">{title}</h2><p className="mt-1 mb-3 text-sm leading-5 text-slate-500">{description}</p>{children}</section>;
}
