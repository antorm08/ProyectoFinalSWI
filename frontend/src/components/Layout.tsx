import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import Icon, { type IconName } from "./Icon";
import { useSystem } from "./SystemContext";

const links: { to: string; label: string; icon: IconName }[] = [
  { to: "/", label: "Dashboard ejecutivo", icon: "grid" },
  { to: "/retrasos", label: "Riesgo de entregas", icon: "shield" },
  { to: "/demanda", label: "Pronóstico de demanda", icon: "chart" },
  { to: "/productos", label: "Clasificación visual", icon: "image" },
  { to: "/vendedores", label: "Segmentación de vendedores", icon: "users" },
  { to: "/resenas", label: "Resumen de reseñas", icon: "text" },
];

function Navigation({ onNavigate }: { onNavigate?: () => void }) {
  const system = useSystem();
  return <>
    <div className="flex items-center gap-2 border-b border-slate-700/60 px-4 py-4"><span className="rounded-lg bg-brand-600 p-2 text-white"><Icon name="layers" className="h-5 w-5" /></span><div><p className="text-base font-bold tracking-tight text-white">CommerceAI</p><p className="mt-0.5 text-xs text-slate-400">Inteligencia comercial y logística</p></div></div>
    <nav aria-label="Navegación principal" className="flex-1 space-y-1 px-3 py-3">
      {links.map((link, i) => <div key={link.to}>
        {(i === 0 || i === 1) && <p className={`px-3 pb-1.5 text-xs font-semibold tracking-wide text-slate-400 uppercase ${i === 1 ? "mt-4" : ""}`}>{i === 0 ? "General" : "Análisis"}</p>}
        <NavLink to={link.to} end={link.to === "/"} onClick={onNavigate} className={({ isActive }) => `flex min-h-11 items-center gap-2 rounded-lg px-3 py-2 text-[13px] font-medium transition-colors ${isActive ? "bg-brand-600 text-white shadow-sm" : "text-slate-300 hover:bg-slate-800 hover:text-white"}`}>
          <Icon name={link.icon} className="h-[18px] w-[18px]" />{link.label}
        </NavLink>
      </div>)}
    </nav>
    <div className="m-3 rounded-lg border border-slate-700 bg-slate-800/60 p-3"><p className="mb-1 text-xs font-semibold text-slate-300">Conexión</p><p className="flex items-center gap-2 text-xs text-slate-300"><Icon name={system.loading ? "refresh" : system.error ? "alert" : "check"} className="h-4 w-4" />{system.loading ? "Comprobando conexión" : system.error ? "Sin conexión" : "Sistema conectado"}</p><button type="button" onClick={system.retry} disabled={system.loading} className="min-h-11 text-xs font-semibold text-blue-300 underline underline-offset-4 disabled:opacity-50">Comprobar conexión</button></div>
  </>;
}

export default function Layout() {
  const [open, setOpen] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const { pathname } = useLocation();
  useEffect(() => {
    if (!open) return;
    const current = dialog.current;
    current?.showModal();
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const media = window.matchMedia("(min-width: 1024px)");
    const closeOnDesktop = () => { if (media.matches) setOpen(false); };
    media.addEventListener("change", closeOnDesktop);
    return () => {
      current?.close();
      document.body.style.overflow = previous;
      media.removeEventListener("change", closeOnDesktop);
      trigger.current?.focus();
    };
  }, [open]);
  useEffect(() => {
    document.title = `${links.find((link) => link.to === pathname)?.label ?? "CommerceAI"} · CommerceAI`;
  }, [pathname]);
  return <div className="min-h-screen">
    <a href="#contenido" className="sr-only fixed top-3 left-3 z-50 rounded-lg bg-white p-3 font-semibold text-brand-700 focus:not-sr-only">Saltar al contenido</a>
    <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 flex-col overflow-y-auto bg-slate-900 lg:flex"><Navigation /></aside>
    <div className="lg:pl-64">
      <header className="flex min-h-15 items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-2 sm:px-6">
        <div className="flex min-w-0 items-center gap-3"><button type="button" ref={trigger} className="btn-secondary p-2.5 lg:hidden" aria-label="Abrir navegación" aria-expanded={open} aria-controls="navegacion-movil" onClick={() => setOpen(true)}><Icon name="menu" /></button><p className="text-sm font-semibold text-slate-800">Centro de operaciones</p></div>
      </header>
      <main id="contenido" tabIndex={-1} className="mx-auto min-w-0 max-w-400 p-4 outline-none sm:px-6 sm:py-5"><Outlet /></main>
      <footer className="mx-4 flex flex-wrap justify-between gap-2 border-t border-slate-200 py-3 text-xs text-slate-500 sm:mx-6"><span>CommerceAI</span><span>BRL · Brasil</span></footer>
    </div>
    {open && <dialog ref={dialog} id="navegacion-movil" aria-label="Navegación de CommerceAI" onCancel={() => setOpen(false)} onClick={(e) => { if (e.target === e.currentTarget) setOpen(false); }} className="fixed inset-0 m-0 h-dvh max-h-none w-full max-w-none bg-transparent p-0 backdrop:bg-slate-950/50">
      <div className="flex h-full w-72 max-w-[90vw] flex-col overflow-y-auto bg-slate-900 text-white"><button type="button" autoFocus onClick={() => setOpen(false)} className="m-3 ml-auto flex min-h-11 items-center gap-2 rounded-lg px-3 text-sm text-slate-200 hover:bg-slate-800"><Icon name="close" />Cerrar</button><Navigation onNavigate={() => setOpen(false)} /></div>
    </dialog>}
  </div>;
}
