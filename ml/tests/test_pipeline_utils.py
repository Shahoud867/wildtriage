from datetime import datetime, timedelta

import numpy as np

from wildtriage_ml.adapters.base import TrackResult
from wildtriage_ml.pipeline.aggregate import aggregate_logits, aggregate_sequence, expand_box
from wildtriage_ml.pipeline.counting import independent_events, max_n, relative_abundance_index
from wildtriage_ml.pipeline.quality import blur_score, brightness, is_night, quality_features
from wildtriage_ml.pipeline.sampling import chunk_ranges, sample_frame_indices


def test_ir_grayscale_frames_are_flagged_as_night():
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 255, size=(32, 32), dtype=np.uint8)
    ir_rgb = np.stack([gray] * 3, axis=-1)
    colour = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    assert is_night(ir_rgb)
    assert not is_night(colour)
    assert set(quality_features(colour)) == {"chroma", "is_night", "brightness", "blur"}


def test_blur_and_brightness():
    rng = np.random.default_rng(1)
    sharp = rng.uniform(0, 1, size=(64, 64))
    smooth = np.tile(np.linspace(0, 1, 64), (64, 1))
    assert blur_score(sharp) > blur_score(smooth)
    assert brightness(np.zeros((8, 8))) == 0.0


def test_max_n_counts_simultaneous_individuals():
    tracks = [
        TrackResult(0, [0, 1, 2, 3], [], "Panthera onca"),
        TrackResult(1, [2, 3, 4], [], "Panthera onca"),
        TrackResult(2, [10, 11], [], "Tapirus bairdii"),
    ]
    assert max_n(tracks) == 2
    assert max_n(tracks, label="Tapirus bairdii") == 1
    assert max_n([]) == 0


def test_independent_events_use_30_minute_rule():
    t0 = datetime(2026, 1, 1, 6, 0)
    recs = [("cam1", "jaguar", t0), ("cam1", "jaguar", t0 + timedelta(minutes=10)),
            ("cam1", "jaguar", t0 + timedelta(minutes=50)), ("cam2", "jaguar", t0)]
    events = independent_events(recs)
    assert events[("cam1", "jaguar")] == 2
    assert events[("cam2", "jaguar")] == 1
    assert relative_abundance_index(2, 50) == 4.0


def test_frame_sampling_and_chunking():
    assert sample_frame_indices(300, 30, 6) == list(range(0, 300, 5))
    assert sample_frame_indices(10, 6, 30) == list(range(10))
    assert chunk_ranges(400, 6, 30) == [(0, 180), (180, 360), (360, 400)]


def test_expand_box_and_aggregate():
    x, y, w, h = expand_box((0.4, 0.4, 0.2, 0.2), 0.5)
    assert np.allclose([x, y, w, h], [0.35, 0.35, 0.3, 0.3])
    assert expand_box((0.0, 0.0, 0.5, 0.5), 1.0)[0] == 0.0
    agg = aggregate_sequence([np.array([0.8, 0.2]), np.array([0.6, 0.4])])
    assert np.allclose(agg, [0.7, 0.3])


def test_logit_averaging_differs_from_probability_averaging_and_is_a_distribution():
    a, b = np.array([4.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])  # one confident frame, one uninformative
    p = aggregate_logits([a, b])
    assert abs(p.sum() - 1) < 1e-12 and p.argmax() == 0
    expected = np.exp([2.0, 0, 0]) / np.exp([2.0, 0, 0]).sum()
    assert np.allclose(p, expected)  # softmax of the mean logits
    from wildtriage_ml.triage.calibration import softmax

    assert not np.allclose(p, aggregate_sequence([softmax(a), softmax(b)]))
    assert np.allclose(aggregate_logits([a + 1000.0, b + 1000.0]), p)  # numerically stable
