"""Unattended supervisor for the lab-PC GPU jobs (ROADMAP §7.3). Run detached; it survives the terminal closing.

    python scripts/lab/pipeline.py            # or: scripts\\lab\\start_pipeline.ps1

Steps run strictly one at a time (two CUDA/PyTorch processes plus Windows and a browser exceed the commit limit
and crash the machine: see docs/lab_pc_notes.md). Every step is resumable and has a validity check, so a failed or
interrupted step is simply retried (up to MAX_TRIES). Progress is written to <STATE_DIR>/pipeline_status.json and a
final report to <STATE_DIR>/PIPELINE_REPORT.md. Safe to start again at any time: finished steps are skipped.
"""

from __future__ import annotations

import ctypes
import glob
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
DATA = Path(os.environ.get("WT_DATA", "D:/wildtriage-data"))
CACHE = Path(os.environ.get("WT_CACHE", "D:/wt-env"))
STATE_DIR = DATA
LOG_DIR = CACHE / "tmp" / "pipeline_logs"
MAX_TRIES = 6
RETRY_WAIT_S = 90

ENV = {
    **os.environ,
    "HF_HOME": str(CACHE / "hf"),
    "HF_TOKEN_PATH": os.environ.get("HF_TOKEN_PATH", str(Path.home() / ".cache" / "huggingface" / "token")),
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "TEMP": str(CACHE / "tmp"),
    "TMP": str(CACHE / "tmp"),
    "PYTHONIOENCODING": "utf-8",
}

status: dict = {"started": datetime.now().isoformat(timespec="seconds"), "steps": {}}


def save_status() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "pipeline_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def keep_awake() -> None:
    """Stop Windows sleeping while jobs run (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    except Exception:
        pass


def commit_gb() -> tuple[float, float]:
    """(committed, limit) in GB from GlobalMemoryStatusEx (ullTotalPageFile is the commit limit)."""

    class MEMSTAT(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong),
                    ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]

    m = MEMSTAT()
    m.l = ctypes.sizeof(MEMSTAT)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return (m.tpf - m.apf) / 2**30, m.tpf / 2**30


def run_command(name: str, cmd: list[str], log: Path) -> int:
    """Run one child, logging its output and the machine's commit level every minute."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} start: {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
        last = 0.0
        while proc.poll() is None:
            time.sleep(5)
            if time.time() - last > 60:
                used, limit = commit_gb()
                fh.write(f"[supervisor] commit {used:.1f}/{limit:.1f} GB\n")
                fh.flush()
                last = time.time()
        return proc.returncode


class Step:
    def __init__(self, name, cmd, check, needs_gpu=True):
        self.name, self.cmd, self.check, self.needs_gpu = name, cmd, check, needs_gpu


# ---- validity checks -------------------------------------------------------------------------------------------

def check_embeddings(split: str):
    def check() -> str | None:
        import numpy as np
        import pandas as pd

        sites = sorted(pd.read_parquet(ROOT / f"research/manifests/lila_{split}_sub.parquet", columns=["site"])["site"].unique())
        missing = []
        for s in sites:
            f = DATA / "embeddings" / f"lila_{split}" / f"{s}.npz"
            try:
                with np.load(f, allow_pickle=False) as z:
                    n = len(z["labels"])
                    ok = n > 200 and z["embeddings"].shape[0] == n and np.isfinite(z["embeddings"]).all()
            except Exception:
                ok = False
            if not ok:
                missing.append(s)
        return f"{len(missing)} of {len(sites)} sites missing/invalid: {missing}" if missing else None

    return check


def check_exp2_run() -> str | None:
    n = len(list((DATA / "exp2" / "clips").glob("*.json")))
    return None if n >= 150 else f"{n}/150 clips cached"


def check_file(path: Path):
    return lambda: None if path.exists() and path.stat().st_size > 0 else f"{path} missing"


STEPS = [
    Step("embed_dev", [PY, "research/data_prep/extract_embeddings.py", "--manifest", "research/manifests/lila_dev_sub.parquet",
                       "--output-dir", str(DATA / "embeddings/lila_dev"), "--batch-size", "64", "--num-workers", "16",
                       "--device", "cuda", "--skip-existing"], check_embeddings("dev")),
    Step("embed_test", [PY, "research/data_prep/extract_embeddings.py", "--manifest", "research/manifests/lila_test_sub.parquet",
                        "--output-dir", str(DATA / "embeddings/lila_test"), "--batch-size", "64", "--num-workers", "16",
                        "--device", "cuda", "--skip-existing"], check_embeddings("test")),
    Step("exp2_run", [PY, "research/experiments/exp2_video_counts.py", "run", "--n-clips", "150", "--out", str(DATA / "exp2")],
         check_exp2_run),
    Step("exp2_eval", [PY, "research/experiments/exp2_video_counts.py", "eval", "--out", str(DATA / "exp2")],
         check_file(DATA / "exp2" / "exp2_summary.json"), needs_gpu=False),
    Step("exp3_depthsam", [PY, "research/experiments/exp3_camouflage.py", "reproduce", "--out", "research/results/exp3_cod10k.json"],
         check_file(ROOT / "research/results/exp3_cod10k.json")),
]


def exp1_step() -> Step:
    out = ROOT / "research/results/exp1_dev_real.txt"
    return Step("exp1_dev", [PY, "-c",
                f"import subprocess,sys; r=subprocess.run([sys.executable,'research/experiments/exp1_triage.py','--sites',r'{DATA}/embeddings/lila_dev/*.npz'],"
                f"capture_output=True,text=True,encoding='utf-8'); open(r'{out}','w',encoding='utf-8').write(r.stdout+r.stderr); sys.exit(r.returncode)"],
                check_file(out), needs_gpu=False)


def wait_for_headroom(need_gb: float, timeout_s: int = 1800) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        used, limit = commit_gb()
        if limit - used >= need_gb:
            return
        time.sleep(30)
    print(f"[supervisor] proceeding with low commit headroom ({commit_gb()})", flush=True)


def main() -> None:
    keep_awake()
    steps = STEPS + [exp1_step()]
    for step in steps:
        info = status["steps"].setdefault(step.name, {"tries": 0, "state": "pending"})
        problem = step.check()
        if problem is None:
            info["state"] = "done (already complete)"
            save_status()
            continue
        for attempt in range(1, MAX_TRIES + 1):
            info.update(state=f"running (try {attempt})", tries=attempt, last_problem=problem)
            save_status()
            if step.needs_gpu:
                wait_for_headroom(12)
            code = run_command(step.name, step.cmd, LOG_DIR / f"{step.name}.log")
            problem = step.check()
            if problem is None:
                info.update(state="done", exit_code=code)
                break
            info.update(state=f"retrying after exit {code}", last_problem=problem)
            save_status()
            time.sleep(RETRY_WAIT_S)
        else:
            info["state"] = f"FAILED: {problem}"
        save_status()
        keep_awake()
    write_report()


def write_report() -> None:
    lines = [f"# Pipeline report ({datetime.now().isoformat(timespec='seconds')})", "", "| step | state | tries |", "|---|---|---|"]
    lines += [f"| {k} | {v['state']} | {v['tries']} |" for k, v in status["steps"].items()]
    try:
        import numpy as np

        lines += ["", "## Embedding sanity (zero-shot top-1 from the cached embeddings; BioCLIP 2, whole images)", ""]
        for f in sorted(glob.glob(str(DATA / "embeddings" / "lila_*" / "*.npz"))):
            with np.load(f, allow_pickle=False) as z:
                acc = float(((z["embeddings"] @ z["text_embeddings"].T).argmax(1) == z["labels"]).mean())
                lines.append(f"- {Path(f).parent.name}/{Path(f).stem}: {len(z['labels'])} images, zero-shot top-1 {acc:.1%}")
    except Exception:
        lines.append(traceback.format_exc())
    for name in ("exp2/exp2_summary.json",):
        p = DATA / name
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
            s.pop("veval_all", None)
            lines += ["", f"## {name}", "```json", json.dumps(s, indent=2), "```"]
    p = ROOT / "research/results/exp3_cod10k.json"
    if p.exists():
        lines += ["", "## DepthSAM COD10K-test", "```json", json.dumps(json.loads(p.read_text(encoding="utf-8"))["results"], indent=2), "```"]
    (STATE_DIR / "PIPELINE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("report written", flush=True)


if __name__ == "__main__":
    main()
