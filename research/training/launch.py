"""Single-GPU launcher for SAM 3 fine-tuning that works on Windows and leaves third_party/sam3 untouched.

    python research/training/launch.py --config stage_a --out D:/wildtriage-data/runs/stage_a \
        --data D:/wildtriage-data/sa_fari_ft [key=value ...]

It composes research/training/configs/<config>.yaml with Hydra (extra key=value arguments override it),
then builds and runs SAM 3's own `Trainer`. Re-running the same command resumes from <out>/checkpoints.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="stage_a")
    ap.add_argument("--out", type=Path, required=True, help="experiment dir: logs, tensorboard, checkpoints")
    ap.add_argument("--data", type=Path, required=True, help="output of research/training/prepare.py")
    ap.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. scratch.max_epochs=2")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.update({"MASTER_ADDR": "localhost", "MASTER_PORT": os.environ.get("MASTER_PORT", "29511"),
                       "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"})

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam3.train.utils.train_utils import makedir, register_omegaconf_resolvers

    import pkg_resources  # noqa: F401  (sam3 resolves its BPE vocabulary through it)
    bpe = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    args.out.mkdir(parents=True, exist_ok=True)
    overrides = [f"paths.data_root={args.data.as_posix()}", f"paths.experiment_log_dir={args.out.as_posix()}",
                 f"paths.bpe_path={Path(bpe).as_posix()}", *args.overrides]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=args.config, overrides=overrides)
    (args.out / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=False), encoding="utf-8")
    makedir(str(args.out))
    register_omegaconf_resolvers()
    instantiate(cfg.trainer, _recursive_=False).run()


if __name__ == "__main__":
    main()
