"""Every Windows script has a macOS/Linux twin, and the shell scripts parse."""

import shutil
import subprocess
from pathlib import Path

import pytest

NATIVE = Path(__file__).resolve().parents[1] / "native"
SHELL_SCRIPTS = sorted(NATIVE.glob("*.sh"))
# Use the bash on PATH explicitly: on Windows, a bare "bash" in subprocess resolves to WSL (System32) first.
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash not installed")


def test_each_powershell_script_has_a_shell_twin():
    ps1 = {p.stem for p in NATIVE.glob("*.ps1")}
    sh = {p.stem for p in SHELL_SCRIPTS}
    assert ps1, "no PowerShell scripts found"
    assert ps1 == sh, f"missing twins: {sorted(ps1 ^ sh)}"


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_script_header_and_line_endings(script):
    raw = script.read_bytes()
    assert b"\r\n" not in raw, "use LF line endings (CRLF breaks bash on macOS/Linux)"
    text = raw.decode("utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text


@needs_bash
@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_script_parses(script):
    # Run by name from the script's folder: Git Bash on Windows cannot open backslash paths.
    result = subprocess.run([BASH, "-n", script.name], cwd=script.parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@needs_bash
def test_mac_worker_refuses_the_gpu_heavy_queue(tmp_path):
    fake_uname = tmp_path / "uname"
    fake_uname.write_text("#!/bin/sh\necho Darwin\n", newline="\n")
    fake_uname.chmod(0o755)
    script = (
        f'PATH="$(cd "{tmp_path.as_posix()}" 2>/dev/null && pwd || echo "{tmp_path.as_posix()}"):$PATH" '
        "bash start-worker.sh gpu,gpu-heavy"
    )
    result = subprocess.run([BASH, "-c", script], cwd=NATIVE, capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    assert "needs CUDA" in result.stderr


@needs_bash
def test_setup_rejects_unknown_machine():
    result = subprocess.run([BASH, "setup.sh", "quantum"], cwd=NATIVE, capture_output=True, text=True)
    assert result.returncode == 2
    assert "usage" in result.stderr
