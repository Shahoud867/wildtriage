import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.training.merge_checkpoint import merge  # noqa: E402


def test_merge_replaces_only_detector_keys_and_keeps_dtype():
    base = {"detector.head.w": torch.zeros(2, 2, dtype=torch.float16), "tracker.mem.w": torch.ones(3)}
    merged, n = merge(base, {"head.w": torch.ones(2, 2)})
    assert n == 1 and merged["detector.head.w"].dtype == torch.float16 and merged["detector.head.w"].sum() == 4
    assert torch.equal(merged["tracker.mem.w"], base["tracker.mem.w"])
    assert base["detector.head.w"].sum() == 0  # the base dict is not modified


def test_merge_refuses_unknown_keys_and_shape_changes():
    base = {"detector.head.w": torch.zeros(2, 2)}
    with pytest.raises(KeyError):
        merge(base, {"nope.w": torch.zeros(2, 2)})
    with pytest.raises(ValueError):
        merge(base, {"head.w": torch.zeros(3, 3)})
