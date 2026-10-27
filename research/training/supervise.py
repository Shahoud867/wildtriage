"""Run launch.py until it exits cleanly, re-launching (and thereby resuming) after a crash.

    python research/training/supervise.py --retries 5 -- --config stage_a --out D:/wildtriage-data/runs/stage_a \
        --data D:/wildtriage-data/sa_fari_ft [key=value ...]

Everything after `--` is passed to launch.py. Each attempt appends to <out>/supervisor.log; <out>/run_info.json
records the git commit, library versions and arguments so the result can be traced to the code that produced it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_info(argv: list[str]) -> dict:
    import torch

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    return {"git_commit": commit or None, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": argv,
            "python": sys.version.split()[0], "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--wait", type=int, default=30, help="seconds between attempts")
    ap.add_argument("--stall", type=int, default=600, help="kill and restart if train.log is silent this long (s)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="environment for the trainer (HF_HOME, TEMP, ...); needed when started detached")
    ap.add_argument("launch_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    env = {**os.environ, **dict(item.split("=", 1) for item in args.env)}
    launch_args = args.launch_args[1:] if args.launch_args[:1] == ["--"] else args.launch_args
    out = Path(launch_args[launch_args.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_info.json").write_text(json.dumps(run_info(launch_args), indent=2), encoding="utf-8")

    for attempt in range(1, args.retries + 2):
        with open(out / "supervisor.log", "a", encoding="utf-8") as log:
            log.write(f"{time.strftime('%F %T')} attempt {attempt}\n")
            log.flush()
            code = run_attempt([sys.executable, "-u", str(HERE / "launch.py"), *launch_args], env, out, args.stall)
            log.write(f"{time.strftime('%F %T')} attempt {attempt} {code}\n")
        if code == 0:
            return
        time.sleep(args.wait)
    sys.exit(1)


def run_attempt(cmd: list[str], env: dict, out: Path, stall_seconds: int) -> str | int:
    """Run one attempt; kill it if train.log stops growing (a hung data loader never exits by itself).

    On Windows a worker that hits the commit limit dies with MemoryError and the trainer then waits forever for
    its batch, so exit-code supervision alone would never restart it.
    """
    log_path = out / "train.log"
    with open(log_path, "a", encoding="utf-8") as stdout, open(out / "train.err", "a", encoding="utf-8") as stderr:
        proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
        started = time.time()
        while proc.poll() is None:
            time.sleep(30)
            if time.time() - max(started, log_path.stat().st_mtime) > stall_seconds:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
                proc.wait()
                return f"stalled (no log output for {stall_seconds} s), killed"
        return proc.returncode


if __name__ == "__main__":
    main()
