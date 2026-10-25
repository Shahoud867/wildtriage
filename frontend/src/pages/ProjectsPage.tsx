import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api/client";

export function ProjectsPage() {
  const qc = useQueryClient();
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.projects });
  const [name, setName] = useState("");
  const create = useMutation({
    mutationFn: () => api.createProject(name),
    onSuccess: () => {
      setName("");
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
  });

  return (
    <div className="mx-auto max-w-3xl p-6">
      <h1 className="mb-4 text-2xl font-semibold">Projects</h1>
      {projects.isError && <p className="text-conf-low">Could not load projects. Are you signed in? <Link className="underline" to="/login">Sign in</Link></p>}
      <ul className="mb-6 divide-y divide-stone-200 rounded border border-stone-200 dark:divide-stone-800 dark:border-stone-800">
        {projects.data?.map((p) => (
          <li key={p.id}>
            <Link to={`/p/${p.id}/overview`} className="block px-4 py-3 hover:bg-stone-100 dark:hover:bg-stone-900">
              <span className="font-medium">{p.name}</span>
              {p.description && <span className="ml-2 text-sm text-stone-500">{p.description}</span>}
            </Link>
          </li>
        ))}
        {projects.data?.length === 0 && <li className="px-4 py-3 text-sm text-stone-500">No projects yet.</li>}
      </ul>
      <form onSubmit={(e) => { e.preventDefault(); create.mutate(); }} className="flex gap-2">
        <input className="flex-1 rounded border px-2 py-1.5 dark:bg-stone-900" placeholder="New project name (admins)"
          value={name} onChange={(e) => setName(e.target.value)} required />
        <button className="rounded bg-brand px-3 py-1.5 text-white" disabled={create.isPending}>Create</button>
      </form>
      {create.isError && <p className="mt-2 text-sm text-conf-low">{create.error.message}</p>}
    </div>
  );
}
