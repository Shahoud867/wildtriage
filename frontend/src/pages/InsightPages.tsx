import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useParams } from "react-router-dom";

import { api } from "../api/client";

function Kpi({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-stone-200 p-4 dark:border-stone-800">
      <div className="text-sm text-stone-500">{label}</div>
      <div className="text-2xl font-semibold">{value}</div>
    </div>
  );
}

export function OverviewPage() {
  const projectId = Number(useParams().projectId);
  const s = useQuery({ queryKey: ["summary", projectId], queryFn: () => api.summary(projectId) });
  const processed = s.data?.media_by_status.processed ?? 0;
  const decisions = Object.values(s.data?.triage ?? {}).reduce((a, b) => a + b, 0);
  const auto = (s.data?.triage.auto_accept ?? 0) + (s.data?.triage.auto_empty ?? 0);
  return (
    <section>
      <h1 className="mb-4 text-2xl font-semibold">Overview</h1>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Kpi label="Media processed" value={processed} />
        <Kpi label="Auto-decided" value={decisions ? `${Math.round((100 * auto) / decisions)}%` : "–"} />
        <Kpi label="Review backlog" value={s.data?.open_reviews ?? "–"} />
        <Kpi label="Species (accepted)" value={Object.keys(s.data?.accepted_species_counts ?? {}).length} />
      </div>
    </section>
  );
}

/** Counts per species. Always says whether numbers include unreviewed predictions. Charts (ECharts, diel activity): W4. */
export function AnalyticsPage() {
  const projectId = Number(useParams().projectId);
  const [acceptedOnly, setAcceptedOnly] = useState(true);
  const s = useQuery({ queryKey: ["summary", projectId], queryFn: () => api.summary(projectId) });
  const counts = (acceptedOnly ? s.data?.accepted_species_counts : s.data?.species_counts) ?? {};
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  return (
    <section className="max-w-2xl">
      <h1 className="mb-4 text-2xl font-semibold">Analytics</h1>
      <label className="mb-3 flex items-center gap-2 text-sm">
        <input type="checkbox" checked={acceptedOnly} onChange={(e) => setAcceptedOnly(e.target.checked)} />
        Accepted only (auto-accepted at the error target, or confirmed by a reviewer)
      </label>
      <table className="w-full text-sm">
        <thead className="text-left text-stone-500"><tr><th className="py-1">Species</th><th>Detections</th></tr></thead>
        <tbody>
          {rows.map(([label, n]) => (
            <tr key={label} className="border-t border-stone-200 dark:border-stone-800"><td className="py-1 italic">{label}</td><td>{n}</td></tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-xs text-stone-500">
        Detection counts are activity indices, not population estimates.
      </p>
    </section>
  );
}

export function ModelsPage() {
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  return (
    <section>
      <h1 className="mb-4 text-2xl font-semibold">Models &amp; Evaluation</h1>
      <table className="w-full text-left text-sm">
        <thead className="text-stone-500"><tr><th className="py-2">Model</th><th>Version</th><th>Task</th><th>License</th><th>Commercial</th><th>Active</th></tr></thead>
        <tbody>
          {models.data?.map((m) => (
            <tr key={m.id} className="border-t border-stone-200 dark:border-stone-800">
              <td className="py-2">{m.name}</td><td>{m.version}</td><td>{m.task}</td><td>{m.license}</td>
              <td>{m.commercial_ok ? "yes" : "no"}</td><td>{m.is_active ? "✓" : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-4 text-sm text-stone-500">Risk–coverage curves, reliability diagrams and confusion matrices: W5.</p>
    </section>
  );
}
