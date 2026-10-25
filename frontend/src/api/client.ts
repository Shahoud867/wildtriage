import type {
  Detection,
  Job,
  Media,
  ModelVersion,
  Project,
  ReviewAction,
  ReviewItem,
  Summary,
  UploadResponse,
  User,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`/api/v1${path}`, { credentials: "include", ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  login: (email: string, password: string) => request<User>("/auth/login", json({ email, password })),
  logout: () => request<void>("/auth/logout", { method: "POST" }),
  me: () => request<User>("/auth/me"),

  projects: () => request<Project[]>("/projects"),
  project: (id: number) => request<Project>(`/projects/${id}`),
  createProject: (name: string, description = "") => request<Project>("/projects", json({ name, description })),
  datasets: (projectId: number) =>
    request<{ id: number; name: string; modality: string }[]>(`/projects/${projectId}/datasets`),
  createDataset: (projectId: number, name: string, modality: string) =>
    request<{ id: number }>("/datasets", json({ project_id: projectId, name, modality })),
  upload: (datasetId: number, files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return request<UploadResponse>(`/datasets/${datasetId}/uploads`, { method: "POST", body: form });
  },

  media: (params: { datasetId?: number; status?: string; limit?: number } = {}) => {
    const q = new URLSearchParams();
    if (params.datasetId) q.set("dataset_id", String(params.datasetId));
    if (params.status) q.set("status", params.status);
    q.set("limit", String(params.limit ?? 100));
    return request<Media[]>(`/media?${q}`);
  },
  mediaItem: (id: number) => request<Media>(`/media/${id}`),
  mediaFileUrl: (id: number) => `/api/v1/media/${id}/file`,
  predictions: (id: number) => request<Detection[]>(`/media/${id}/predictions`),

  jobs: (projectId: number) => request<Job[]>(`/jobs?project_id=${projectId}`),
  job: (id: number) => request<Job>(`/jobs/${id}`),
  cancelJob: (id: number) => request<Job>(`/jobs/${id}/cancel`, { method: "POST" }),

  reviewQueue: (projectId: number) => request<ReviewItem[]>(`/review/queue?project_id=${projectId}`),
  decide: (taskId: number, action: ReviewAction, extra: { new_label?: string; time_spent_ms?: number } = {}) =>
    request(`/review/tasks/${taskId}/decision`, json({ action, ...extra })),

  models: () => request<ModelVersion[]>("/models"),
  summary: (projectId: number) => request<Summary>(`/analytics/${projectId}/summary`),
};
