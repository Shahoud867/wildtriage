"""Loads models on demand and keeps total accelerator memory under a per-machine budget.

Lab PC (RTX 4000 Ada, 20 GB) -> 18 GB, bf16. T1000 PC (4 GB, Turing) -> 3.5 GB, fp16.
MacBook Pro M1 Pro (32 GB unified memory, MPS) -> e.g. 12 GB, fp32, leaving room for the OS.
Least-recently-used models are unloaded first, so heavy experts (SAM 3, DepthSAM, a VLM) never sit in
memory together. A model that can never fit, or cannot run on this device, is refused at load time
instead of crashing mid-job.
"""

from collections import OrderedDict
from collections.abc import Callable

from wildtriage_ml.adapters.base import Adapter
from wildtriage_ml.device import device_kind, empty_cache, pick_device, pick_dtype

__all__ = ["DeviceNotSupportedError", "ModelRegistry", "VRAMBudgetError", "pick_dtype"]


class VRAMBudgetError(RuntimeError):
    pass


class DeviceNotSupportedError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, vram_budget_gb: float, device: str | None = None):
        self.vram_budget_gb = vram_budget_gb
        self.device = pick_device(device)
        self._loaded: OrderedDict[str, Adapter] = OrderedDict()

    @property
    def used_gb(self) -> float:
        return sum(a.info.vram_gb for a in self._loaded.values())

    @property
    def loaded(self) -> list[str]:
        return list(self._loaded)

    def get(self, key: str, factory: Callable[[], Adapter]) -> Adapter:
        if key in self._loaded:
            self._loaded.move_to_end(key)
            return self._loaded[key]
        adapter = factory()
        if device_kind(self.device) not in adapter.info.devices:
            raise DeviceNotSupportedError(
                f"{key} runs on {', '.join(adapter.info.devices)}, not on {self.device}; "
                "route it to the gpu-heavy queue (lab PC, or molab)"
            )
        need = adapter.info.vram_gb
        if need > self.vram_budget_gb:
            raise VRAMBudgetError(
                f"{key} needs ~{need} GB, more than this machine's {self.vram_budget_gb} GB budget; "
                "route it to the gpu-heavy queue (lab PC, or molab)"
            )
        while self._loaded and self.used_gb + need > self.vram_budget_gb:
            self.evict(next(iter(self._loaded)))
        adapter.load(self.device)
        self._loaded[key] = adapter
        return adapter

    def evict(self, key: str) -> None:
        adapter = self._loaded.pop(key, None)
        if adapter is None:
            return
        adapter.unload()
        empty_cache(self.device)

    def clear(self) -> None:
        for key in list(self._loaded):
            self.evict(key)
