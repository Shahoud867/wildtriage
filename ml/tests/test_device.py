"""Device selection (cuda -> mps -> cpu), precision per device, and cache clearing.

PyTorch is not installed in CI, so these tests use a small fake `torch` module.
"""

import sys
from types import SimpleNamespace

import pytest

from wildtriage_ml.adapters.base import ModelInfo
from wildtriage_ml.device import empty_cache, pick_device, pick_dtype
from wildtriage_ml.registry import DeviceNotSupportedError, ModelRegistry
from wildtriage_ml.runtime import Pipeline


def fake_torch(cuda=False, mps=False, bf16=True):
    calls = {"cuda_empty": 0, "mps_empty": 0}

    def bump(key):
        calls[key] += 1

    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, is_bf16_supported=lambda: bf16,
                             empty_cache=lambda: bump("cuda_empty")),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        mps=SimpleNamespace(empty_cache=lambda: bump("mps_empty")),
        float32="float32", float16="float16", bfloat16="bfloat16",
        calls=calls,
    )


@pytest.fixture(autouse=True)
def no_device_env(monkeypatch):
    monkeypatch.delenv("WT_DEVICE", raising=False)


def use(monkeypatch, torch):
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    use(monkeypatch, fake_torch(cuda=cuda, mps=mps))
    assert pick_device() == expected
    assert pick_device("auto") == expected


def test_without_torch_auto_means_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    assert pick_device() == "cpu"
    with pytest.raises(RuntimeError, match="not installed"):
        pick_device("mps")


def test_env_var_and_explicit_choice(monkeypatch):
    use(monkeypatch, fake_torch(cuda=True, mps=True))
    monkeypatch.setenv("WT_DEVICE", "mps")
    assert pick_device() == "mps"  # env var pins it
    assert pick_device("cpu") == "cpu"  # an explicit argument wins over the env var
    assert pick_device("cuda:1") == "cuda:1"


def test_unavailable_or_unknown_devices_fail_loudly(monkeypatch):
    use(monkeypatch, fake_torch(cuda=False, mps=False))
    with pytest.raises(RuntimeError, match="CUDA requested"):
        pick_device("cuda")
    with pytest.raises(RuntimeError, match="MPS requested"):
        pick_device("mps")
    with pytest.raises(ValueError):
        pick_device("tpu")
    with pytest.raises(ValueError):
        pick_device("mps:0")


def test_precision_per_device(monkeypatch):
    use(monkeypatch, fake_torch(cuda=True, bf16=True))
    assert pick_dtype("cuda") == "bfloat16"  # lab PC (Ada)
    use(monkeypatch, fake_torch(cuda=True, bf16=False))
    assert pick_dtype("cuda:0") == "float16"  # T1000 (Turing)
    assert pick_dtype("mps") == "float32"  # MacBook M1 Pro: no hardware bf16
    assert pick_dtype("cpu") == "float32"


def test_cache_clearing_targets_the_right_backend(monkeypatch):
    torch = use(monkeypatch, fake_torch(cuda=True, mps=True))
    empty_cache("mps")
    empty_cache("cuda")
    empty_cache("cpu")
    assert torch.calls == {"cuda_empty": 1, "mps_empty": 1}


class FakeModel:
    def __init__(self, name, devices=("cuda", "mps", "cpu")):
        self.info = ModelInfo(name, "0", "detect", "MIT", True, vram_gb=1.0, devices=devices)

    def load(self, device):
        self.device = device

    def unload(self):
        pass


def test_registry_on_mps_loads_portable_models_and_clears_mps_cache(monkeypatch):
    torch = use(monkeypatch, fake_torch(mps=True))
    reg = ModelRegistry(vram_budget_gb=1.5)  # auto -> mps
    assert reg.device == "mps"
    model = reg.get("bioclip", lambda: FakeModel("bioclip"))
    assert model.device == "mps"
    reg.get("megadetector", lambda: FakeModel("megadetector"))  # evicts bioclip (budget)
    assert torch.calls["mps_empty"] == 1


def test_registry_refuses_cuda_only_models_on_a_mac(monkeypatch):
    use(monkeypatch, fake_torch(mps=True))
    reg = ModelRegistry(vram_budget_gb=12)
    with pytest.raises(DeviceNotSupportedError, match="gpu-heavy"):
        reg.get("sam3", lambda: FakeModel("sam3", devices=("cuda",)))
    assert reg.loaded == []


def test_real_adapters_declare_where_they_run():
    from wildtriage_ml.adapters.experimental import DepthSAMRefiner, OWLAerialCounter
    from wildtriage_ml.adapters.sam3 import Sam3VideoTracker

    assert Sam3VideoTracker.info.devices == ("cuda",)
    assert DepthSAMRefiner.info.devices == ("cuda",)
    assert "mps" not in OWLAerialCounter.info.devices


def test_pipeline_device(monkeypatch):
    use(monkeypatch, fake_torch(cuda=False, mps=True))
    assert Pipeline(backend="stub").registry.device == "cpu"  # stubs always on CPU
    assert Pipeline(backend="real").registry.device == "mps"
    assert Pipeline(backend="real", device="cpu").registry.device == "cpu"
