"""Counts and activity summaries. These are activity indices, not population (density) estimates."""

from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta

from wildtriage_ml.adapters.base import TrackResult


def max_n(tracks: Iterable[TrackResult], label: str | None = None) -> int:
    """MaxN: the most individuals visible in any single frame of a clip (optionally for one label)."""
    per_frame: Counter[int] = Counter()
    for t in tracks:
        if label is not None and t.label != label:
            continue
        for f in set(t.frames):
            per_frame[f] += 1
    return max(per_frame.values(), default=0)


def independent_events(
    records: Iterable[tuple[str, str, datetime]], gap: timedelta = timedelta(minutes=30)
) -> dict[tuple[str, str], int]:
    """Count independent detection events per (deployment, species).

    A new event starts when the same species is seen at the same deployment more than `gap`
    after its previous detection (the common 30-minute independence rule).
    """
    by_key: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    for deployment, species, ts in records:
        by_key[(deployment, species)].append(ts)
    events = {}
    for key, times in by_key.items():
        times.sort()
        n, last = 0, None
        for ts in times:
            if last is None or ts - last > gap:
                n += 1
            last = ts
        events[key] = n
    return events


def relative_abundance_index(n_events: int, trap_days: float) -> float:
    """Events per 100 trap-days. Not comparable across species with different detectability."""
    return 100.0 * n_events / trap_days if trap_days > 0 else float("nan")
