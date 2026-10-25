// Hand-written until the API stabilises; then replace with `npm run gen:api` (openapi-typescript).

export type Role = "admin" | "reviewer" | "viewer";
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type ReviewAction = "confirm" | "relabel" | "mark_empty" | "reject_box" | "adjust_box" | "unsure";

export interface User {
  id: number;
  email: string;
  name: string;
  role: Role;
}

export interface Project {
  id: number;
  name: string;
  description: string;
  settings: Record<string, unknown>;
}

export interface UploadResponse {
  results: { filename: string; ok: boolean; media_id: number | null; code: string | null; message: string | null }[];
  job_id: number | null;
}

export interface Media {
  id: number;
  dataset_id: number;
  kind: "image" | "video" | "aerial_image";
  original_filename: string;
  width: number | null;
  height: number | null;
  is_night: boolean | null;
  quality: Record<string, number | boolean>;
  status: string;
  error: string | null;
}

export interface Detection {
  id: number;
  category: string;
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  label: string | null;
  calibrated_score: number | null;
  topk: { label: string; p: number }[];
}

export interface Job {
  id: number;
  type: string;
  queue: string;
  status: JobStatus;
  progress: number;
  total_items: number;
  done_items: number;
  error: string | null;
  created_at: string | null;
}

export interface ReviewItem {
  task_id: number;
  media_id: number;
  detection_id: number | null;
  priority: number;
  reasons: string[];
  label: string | null;
  calibrated_score: number | null;
  topk: { label: string; p: number }[];
}

export interface ModelVersion {
  id: number;
  name: string;
  version: string;
  task: string;
  license: string;
  commercial_ok: boolean;
  is_active: boolean;
}

export interface Summary {
  media_by_status: Record<string, number>;
  triage: Record<string, number>;
  open_reviews: number;
  species_counts: Record<string, number>;
  accepted_species_counts: Record<string, number>;
}
