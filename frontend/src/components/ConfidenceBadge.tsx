import { confidenceLevel, formatPercent } from "../lib/confidence";

const STYLE = {
  high: { cls: "bg-conf-high/15 text-conf-high", icon: "●", text: "High" },
  medium: { cls: "bg-conf-med/15 text-conf-med", icon: "◐", text: "Medium" },
  low: { cls: "bg-conf-low/15 text-conf-low", icon: "○", text: "Low" },
};

/** Calibrated confidence: always number + icon + word, never colour alone. */
export function ConfidenceBadge({ p }: { p: number | null | undefined }) {
  if (p == null) return <span className="text-sm text-stone-500">–</span>;
  const s = STYLE[confidenceLevel(p)];
  return (
    <span className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-sm font-medium ${s.cls}`}>
      <span aria-hidden>{s.icon}</span>
      {formatPercent(p)} <span className="sr-only">confidence,</span>
      <span className="font-normal">{s.text}</span>
    </span>
  );
}
