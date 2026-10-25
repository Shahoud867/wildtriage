import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";

import { api } from "../api/client";
import type { ReviewAction } from "../api/types";
import { ConfidenceBadge } from "../components/ConfidenceBadge";
import { useHotkeys } from "../hooks/useHotkeys";
import { reasonText } from "../lib/confidence";

/** Risk-ordered queue, one item at a time. Keys: A accept · 1–5 pick top-k · E empty · X reject box · U unsure. */
export function ReviewPage() {
  const projectId = Number(useParams().projectId);
  const qc = useQueryClient();
  const queue = useQuery({ queryKey: ["review", projectId], queryFn: () => api.reviewQueue(projectId) });
  const item = queue.data?.[0];
  const shownAt = useRef(performance.now());
  const [done, setDone] = useState(0);

  const decide = useMutation({
    mutationFn: ({ action, label }: { action: ReviewAction; label?: string }) =>
      api.decide(item!.task_id, action, {
        new_label: label,
        time_spent_ms: Math.round(performance.now() - shownAt.current),
      }),
    onSuccess: () => {
      setDone((n) => n + 1);
      shownAt.current = performance.now();
      qc.invalidateQueries({ queryKey: ["review", projectId] });
    },
  });

  const bindings = useMemo(() => {
    const b: Record<string, () => void> = {
      a: () => decide.mutate({ action: "confirm" }),
      e: () => decide.mutate({ action: "mark_empty" }),
      x: () => decide.mutate({ action: "reject_box" }),
      u: () => decide.mutate({ action: "unsure" }),
    };
    item?.topk.slice(0, 5).forEach((t, i) => {
      b[String(i + 1)] = () =>
        decide.mutate(t.label === item.label ? { action: "confirm" } : { action: "relabel", label: t.label });
    });
    return b;
  }, [item, decide]);
  useHotkeys(bindings, !!item && !decide.isPending);

  if (queue.isLoading) return <p>Loading…</p>;
  if (!item) return <p className="text-stone-500">Queue is empty. {done > 0 && `You reviewed ${done} items.`}</p>;

  return (
    <section className="grid gap-6 lg:grid-cols-[1fr_340px]">
      <img src={api.mediaFileUrl(item.media_id)} alt="item under review" className="w-full rounded" />
      <aside className="space-y-4 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-stone-500">{queue.data!.length} in queue · {done} done</span>
          <span className="text-stone-500">risk {item.priority.toFixed(2)}</span>
        </div>
        <div>
          <p className="mb-1 font-medium">Why this needs review</p>
          <ul className="flex flex-wrap gap-1">
            {item.reasons.map((r) => <li key={r} className="rounded bg-stone-200 px-2 py-0.5 dark:bg-stone-800">{reasonText(r)}</li>)}
          </ul>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-lg font-semibold">{item.label ?? "No species prediction"}</span>
          <ConfidenceBadge p={item.calibrated_score} />
        </div>
        <ol className="space-y-1">
          {item.topk.slice(0, 5).map((t, i) => (
            <li key={t.label}>
              <button className="flex w-full justify-between rounded border px-3 py-1.5 hover:bg-stone-100 dark:border-stone-800 dark:hover:bg-stone-900"
                onClick={() => bindings[String(i + 1)]()}>
                <span><kbd className="mr-2 text-stone-500">{i + 1}</kbd>{t.label}</span>
                <span>{Math.round(t.p * 100)}%</span>
              </button>
            </li>
          ))}
        </ol>
        <div className="grid grid-cols-2 gap-2">
          <button className="rounded bg-brand px-3 py-2 text-white" onClick={bindings.a}><kbd>A</kbd> Accept</button>
          <button className="rounded border px-3 py-2 dark:border-stone-800" onClick={bindings.e}><kbd>E</kbd> Empty</button>
          <button className="rounded border px-3 py-2 dark:border-stone-800" onClick={bindings.x}><kbd>X</kbd> Reject box</button>
          <button className="rounded border px-3 py-2 dark:border-stone-800" onClick={bindings.u}><kbd>U</kbd> Unsure</button>
        </div>
        {decide.isError && <p className="text-conf-low">{decide.error.message}</p>}
      </aside>
    </section>
  );
}
