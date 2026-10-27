import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
pytest.importorskip("torch")
pytest.importorskip("pandas")
from extract_embeddings import is_valid_npz  # noqa: E402


def _write(path, **overrides):
    data = dict(embeddings=np.ones((300, 8), np.float32), labels=np.zeros(300, np.int64),
                interval=np.zeros(300, np.int64), is_night=np.zeros(300, bool),
                text_embeddings=np.ones((3, 8), np.float32), logit_scale=30.0)
    data.update(overrides)
    np.savez_compressed(path, **data)


def test_valid_file_with_scalar_logit_scale_is_accepted(tmp_path):
    f = tmp_path / "a.npz"
    _write(f)
    assert is_valid_npz(f)  # regression: len() of the 0-d logit_scale used to make this False


def test_missing_key_or_empty_array_is_rejected(tmp_path):
    f = tmp_path / "b.npz"
    _write(f, embeddings=np.zeros((0, 8), np.float32))
    assert not is_valid_npz(f)
    assert not is_valid_npz(tmp_path / "missing.npz")
