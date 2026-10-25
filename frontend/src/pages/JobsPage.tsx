import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";

import { api } from "../api/client";

/** Polls every 2 s while any job is active. W2 replaces polling with SSE (GET /jobs/{id}/events). */
export function JobsPage() {
  const projectId = Number(useParams().projectId);
  const jobs = useQuery({
    queryKey: ["jobs", projectId],
    queryFn: () => api.jobs(projectId),
    refetchInterval: (q) => (q.state.data?.some((j) => j.status === "queued" || j.status === "running") ? 2000 : false),
  });

  return (
    <section>
      <h1 className="mb-4 text-2xl font-semibold">Jobs</h1>
      <table className="w-full text-left text-sm">
        <thead className="text-stone-500">
          <tr><th className="py-2">#</th><th>Type</th><th>Queue</th><th>Status</th><th className="w-1/3">Progress</th><th>Error</th></tr>
        </thead>
        <tbody>
          {jobs.data?.map((j) => (
            <tr key={j.id} className="border-t border-stone-200 dark:border-stone-800">
              <td className="py-2">{j.id}</td>
              <td>{j.type}</td>
              <td><code>{j.queue}</code></td>
              <td>{j.status}</td>
              <td>
                <div className="h-2 rounded bg-stone-200 dark:bg-stone-800" role="progressbar" aria-valuenow={Math.round(j.progress * 100)}>
                  <div className="h-2 rounded bg-brand" style={{ width: `${Math.round(j.progress * 100)}%` }} />
                </div>
                <span className="text-xs text-stone-500">{j.done_items}/{j.total_items}</span>
              </td>
              <td className="max-w-xs truncate text-conf-low" title={j.error ?? ""}>{j.error?.split("\n").at(-2) ?? ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {jobs.data?.length === 0 && <p className="mt-4 text-sm text-stone-500">No jobs yet. Upload media on the Data page.</p>}
    </section>
  );
}
