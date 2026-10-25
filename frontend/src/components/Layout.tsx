import type { ReactNode } from "react";
import { NavLink, Outlet, useParams } from "react-router-dom";

const NAV = [
  ["overview", "Overview"],
  ["data", "Data"],
  ["jobs", "Jobs"],
  ["media", "Media"],
  ["review", "Review"],
  ["analytics", "Analytics"],
  ["models", "Models & Evaluation"],
  ["exports", "Exports"],
  ["settings", "Settings"],
] as const;

export function Layout() {
  const { projectId } = useParams();
  return (
    <div className="flex min-h-screen">
      <nav className="w-56 shrink-0 border-r border-stone-200 p-4 dark:border-stone-800">
        <NavLink to="/projects" className="mb-6 block text-lg font-semibold text-brand">
          WildTriage
        </NavLink>
        <ul className="space-y-1">
          {NAV.map(([path, label]) => (
            <li key={path}>
              <NavLink
                to={`/p/${projectId}/${path}`}
                className={({ isActive }) =>
                  `block rounded px-3 py-1.5 text-sm ${isActive ? "bg-brand text-white" : "hover:bg-stone-200 dark:hover:bg-stone-800"}`
                }
              >
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      <main className="min-w-0 flex-1 p-6">
        <Outlet />
      </main>
    </div>
  );
}

export function Placeholder({ title, week, children }: { title: string; week: string; children?: ReactNode }) {
  return (
    <section>
      <h1 className="mb-2 text-2xl font-semibold">{title}</h1>
      <p className="mb-4 text-sm text-stone-500">Planned for {week} (see docs/PROJECT_PLAN.md §10).</p>
      {children}
    </section>
  );
}
