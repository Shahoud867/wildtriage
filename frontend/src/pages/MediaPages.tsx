import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";

import { api } from "../api/client";
import { ConfidenceBadge } from "../components/ConfidenceBadge";

/** Thumbnail grid. W2-W4: filters (species, triage status, confidence range, day/night, deployment) + virtualisation. */
export function MediaPage() {
  const projectId = Number(useParams().projectId);
  const media = useQuery({ queryKey: ["media"], queryFn: () => api.media({ limit: 200 }) });
  return (
    <section>
      <h1 className="mb-4 text-2xl font-semibold">Media</h1>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-3">
        {media.data?.map((m) => (
          <Link key={m.id} to={`/p/${projectId}/media/${m.id}`}
            className="overflow-hidden rounded border border-stone-200 text-xs dark:border-stone-800">
            {m.kind === "video" ? (
              <div className="flex aspect-video items-center justify-center bg-stone-200 dark:bg-stone-800">▶ video</div>
            ) : (
              <img src={api.mediaFileUrl(m.id)} alt={m.original_filename} loading="lazy" className="aspect-video w-full object-cover" />
            )}
            <div className="flex justify-between p-2">
              <span className="truncate">{m.original_filename}</span>
              <span>{m.is_night ? "🌙" : ""} {m.status}</span>
            </div>
          </Link>
        ))}
      </div>
    </section>
  );
}

/** One image with box overlays (normalised coords -> percentages). Video overlays and masks: W3. */
export function MediaDetailPage() {
  const mediaId = Number(useParams().mediaId);
  const media = useQuery({ queryKey: ["media", mediaId], queryFn: () => api.mediaItem(mediaId) });
  const preds = useQuery({ queryKey: ["predictions", mediaId], queryFn: () => api.predictions(mediaId) });
  if (!media.data) return <p>Loading…</p>;
  return (
    <section className="grid gap-6 lg:grid-cols-[1fr_320px]">
      <div className="relative self-start">
        <img src={api.mediaFileUrl(mediaId)} alt={media.data.original_filename} className="w-full rounded" />
        {preds.data?.map((d) => (
          <div key={d.id} className="absolute border-2 border-yellow-400"
            style={{ left: `${d.x * 100}%`, top: `${d.y * 100}%`, width: `${d.w * 100}%`, height: `${d.h * 100}%` }}>
            <span className="absolute -top-6 left-0 whitespace-nowrap bg-yellow-400 px-1 text-xs text-black">
              {d.label ?? d.category}
            </span>
          </div>
        ))}
      </div>
      <aside className="space-y-4 text-sm">
        <h1 className="text-lg font-semibold">{media.data.original_filename}</h1>
        <p>{media.data.is_night ? "Night / IR" : "Day"} · status {media.data.status}</p>
        {preds.data?.length === 0 && <p className="text-stone-500">No detections (empty candidate).</p>}
        {preds.data?.map((d) => (
          <div key={d.id} className="rounded border p-3 dark:border-stone-800">
            <div className="mb-2 flex items-center justify-between">
              <span className="font-medium">{d.label ?? d.category}</span>
              <ConfidenceBadge p={d.calibrated_score} />
            </div>
            <ol className="space-y-0.5 text-stone-600 dark:text-stone-400">
              {d.topk.map((t) => <li key={t.label} className="flex justify-between"><span>{t.label}</span><span>{Math.round(t.p * 100)}%</span></li>)}
            </ol>
            <p className="mt-2 text-xs text-stone-500">detector score {d.score.toFixed(2)}</p>
          </div>
        ))}
      </aside>
    </section>
  );
}
