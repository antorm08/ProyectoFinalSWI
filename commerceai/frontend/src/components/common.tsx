import type { ReactNode } from "react";

export function PageHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <header className="mb-6">
      <h2 className="text-2xl font-bold">{title}</h2>
      <p className="text-slate-500">{subtitle}</p>
    </header>
  );
}

export function ModelBadge({ trained }: { trained: boolean }) {
  return trained ? (
    <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs text-emerald-700">Modelo entrenado</span>
  ) : (
    <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-700">
      Modo provisional (modelo sin entrenar)
    </span>
  );
}

export function ErrorBox({ children }: { children: ReactNode }) {
  return children ? <div className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{children}</div> : null;
}
