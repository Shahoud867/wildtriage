import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";

import { api } from "../api/client";
import type { UploadResponse } from "../api/types";

/** Datasets and upload. Files are validated by content on the server; results are shown per file. */
export function DataPage() {
  const projectId = Number(useParams().projectId);
  const qc = useQueryClient();
  const datasets = useQuery({ queryKey: ["datasets", projectId], queryFn: () => api.datasets(projectId) });
  const [datasetId, setDatasetId] = useState<number | null>(null);
  const [newName, setNewName] = useState("");
  const [modality, setModality] = useState("camera_trap");
  const [result, setResult] = useState<UploadResponse | null>(null);

  const createDataset = useMutation({
    mutationFn: () => api.createDataset(projectId, newName, modality),
    onSuccess: (d) => {
      setDatasetId(d.id);
      setNewName("");
      qc.invalidateQueries({ queryKey: ["datasets", projectId] });
    },
  });
  const upload = useMutation({
    mutationFn: (files: File[]) => api.upload(datasetId!, files),
    onSuccess: setResult,
  });

  return (
    <section className="max-w-4xl space-y-6">
      <h1 className="text-2xl font-semibold">Data</h1>
      <div className="flex flex-wrap items-end gap-2">
        <label className="text-sm">
          Dataset
          <select className="ml-2 rounded border px-2 py-1.5 dark:bg-stone-900" value={datasetId ?? ""}
            onChange={(e) => setDatasetId(Number(e.target.value) || null)}>
            <option value="">Choose…</option>
            {datasets.data?.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.modality})</option>)}
          </select>
        </label>
        <span className="text-sm text-stone-500">or</span>
        <input className="rounded border px-2 py-1.5 dark:bg-stone-900" placeholder="New dataset (e.g. SD card 01)"
          value={newName} onChange={(e) => setNewName(e.target.value)} />
        <select className="rounded border px-2 py-1.5 dark:bg-stone-900" value={modality} onChange={(e) => setModality(e.target.value)}>
          <option value="camera_trap">Camera trap</option>
          <option value="aerial">Aerial (stills only, experimental)</option>
        </select>
        <button className="rounded bg-brand px-3 py-1.5 text-white disabled:opacity-50" disabled={!newName}
          onClick={() => createDataset.mutate()}>Create</button>
      </div>

      <label className={`block rounded-lg border-2 border-dashed p-8 text-center ${datasetId ? "cursor-pointer" : "opacity-50"}`}>
        <span className="text-sm">Drop or choose images/videos (JPG, PNG, TIFF, MP4, MOV, AVI)</span>
        <input type="file" multiple className="sr-only" disabled={!datasetId || upload.isPending}
          accept=".jpg,.jpeg,.png,.tif,.tiff,.mp4,.mov,.avi"
          onChange={(e) => e.target.files && upload.mutate(Array.from(e.target.files))} />
      </label>
      {upload.isPending && <p className="text-sm">Uploading…</p>}
      {upload.isError && <p className="text-sm text-conf-low">{upload.error.message}</p>}

      {result && (
        <div>
          <p className="mb-2 text-sm">
            {result.results.filter((r) => r.ok).length} accepted, {result.results.filter((r) => !r.ok).length} rejected.
            {result.job_id && <> Processing started: <Link className="underline" to={`/p/${projectId}/jobs`}>job #{result.job_id}</Link>.</>}
          </p>
          <ul className="max-h-64 overflow-auto rounded border text-sm dark:border-stone-800">
            {result.results.map((r, i) => (
              <li key={i} className="flex justify-between border-b px-3 py-1 last:border-0 dark:border-stone-800">
                <span>{r.filename}</span>
                <span className={r.ok ? "text-conf-high" : "text-conf-low"}>{r.ok ? "✓ accepted" : `✗ ${r.code}: ${r.message}`}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <p className="text-sm text-stone-500">Deployments (sites, cameras, dates) and dataset version snapshots: W1–W2.</p>
    </section>
  );
}
