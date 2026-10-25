import { useEffect } from "react";

/** Keyboard-first review: maps single keys to handlers, ignoring keys typed into inputs. */
export function useHotkeys(bindings: Record<string, () => void>, enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.closest("input, textarea, select, [contenteditable]") || e.ctrlKey || e.metaKey || e.altKey) return;
      const fn = bindings[e.key.toLowerCase()];
      if (fn) {
        e.preventDefault();
        fn();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [bindings, enabled]);
}
