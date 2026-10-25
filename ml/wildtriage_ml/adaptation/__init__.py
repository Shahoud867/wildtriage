"""Per-site adaptation from verified labels, following Paper 3's (StreamTrap, ECCV 2026) recipe:
Balanced Softmax + a lightweight update (linear head here; LoRA in lora.py), then WiSE interpolation
back toward the zero-shot head."""
