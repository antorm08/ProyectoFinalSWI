import { NavLink, Outlet } from "react-router-dom";

const links = [
  { to: "/", label: "Panel general" },
  { to: "/retrasos", label: "Riesgo de retraso" },
  { to: "/demanda", label: "Demanda" },
  { to: "/productos", label: "Productos (imagen)" },
  { to: "/vendedores", label: "Vendedores" },
  { to: "/resenas", label: "Reseñas" },
];

export default function Layout() {
  return (
    <div className="flex min-h-screen">
      <aside className="w-60 shrink-0 bg-slate-900 p-5 text-slate-200">
        <h1 className="mb-8 text-xl font-bold text-white">CommerceAI</h1>
        <nav className="flex flex-col gap-1">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.to === "/"}
              className={({ isActive }) =>
                `rounded-lg px-3 py-2 text-sm ${isActive ? "bg-indigo-600 text-white" : "hover:bg-slate-800"}`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="flex-1 p-8">
        <Outlet />
      </main>
    </div>
  );
}
