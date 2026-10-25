export type ConfidenceLevel = "high" | "medium" | "low";

/** Map a *calibrated* probability to a display level. Thresholds are for display only, not for triage. */
export function confidenceLevel(p: number): ConfidenceLevel {
  if (p >= 0.9) return "high";
  if (p >= 0.6) return "medium";
  return "low";
}

export function formatPercent(p: number | null | undefined): string {
  return p == null ? "–" : `${Math.round(p * 100)}%`;
}

const REASON_TEXT: Record<string, string> = {
  low_confidence: "Low confidence",
  night: "Night / IR",
  ambiguous_species: "Ambiguous species",
  tiny_object: "Tiny animal",
  too_dark: "Too dark",
  blurred: "Blurred",
  possible_missed_animal: "Possible missed animal",
  audit_sample: "Random audit",
  video_track_review: "Video tracks",
};

export function reasonText(code: string): string {
  return REASON_TEXT[code] ?? code.replaceAll("_", " ");
}
