def sample_frame_indices(n_frames: int, src_fps: float, target_fps: float) -> list[int]:
    """Frame indices that resample a clip to `target_fps` (1 fps screening, 6 fps tracking)."""
    if n_frames <= 0 or src_fps <= 0 or target_fps <= 0:
        return []
    if target_fps >= src_fps:
        return list(range(n_frames))
    step = src_fps / target_fps
    indices, t = [], 0.0
    while round(t) < n_frames:
        idx = int(round(t))
        if not indices or idx != indices[-1]:
            indices.append(idx)
        t += step
    return indices


def chunk_ranges(n_frames: int, fps: float, max_chunk_s: float) -> list[tuple[int, int]]:
    """Split [0, n_frames) into chunks of at most `max_chunk_s` seconds (keeps SAM 3 within VRAM)."""
    size = max(1, int(fps * max_chunk_s))
    return [(start, min(start + size, n_frames)) for start in range(0, n_frames, size)]
