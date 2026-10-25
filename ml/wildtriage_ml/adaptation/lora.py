"""LoRA on BioCLIP 2's vision tower (P1, W5). StreamTrap reports LoRA + Balanced Softmax as its best
model-update recipe; we start with linear heads (heads.py) and add LoRA only for the ablation.

Plan: wrap the OpenCLIP visual transformer's attention projections with `peft.LoraConfig`
(rank 8-16), train with Balanced Softmax on verified crops of one site, on the lab PC (bf16),
and compare against the linear head at equal label budgets. Too heavy for the 4 GB T1000.
"""


def train_lora(*args, **kwargs):
    raise NotImplementedError("W5: LoRA ablation (see module docstring)")
