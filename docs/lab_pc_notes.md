# University PC notes (RTX 4000 Ada)

Findings from the first session, 26 Oct 2026. Complements ROADMAP §7.

## Memory: run one GPU process at a time
- The machine has 31.7 GB RAM and a pagefile on D:, giving a **commit limit of about 36–39 GB**. Windows, Chrome and
  VS Code already commit about 20 GB.
- One PyTorch/CUDA process commits **8–10 GB** (BioCLIP 2 extraction, DepthSAM) and SAM 3 about **15–18 GB**,
  though its working set is only 4–8 GB.
- Two or three such processes exceed the limit. Windows logs event 2004 ("low virtual memory"), python dies with
  "CUDA error: out of memory" or an access violation, and VS Code and the desktop session can crash and reload.
  This, not GPU memory (20 GB), was the cause of the crashes.
- So `scripts/lab/pipeline.py` runs its steps strictly one at a time. Do not start a second GPU job by hand while
  it runs. Close the browser if memory is tight.
- If admin access ever becomes available: enlarge the pagefile (System Properties > Advanced > Performance >
  Virtual memory, e.g. 64 GB on D:).

## Environment on this PC
- No git, no admin. uv and Python 3.12 live in `D:\wt-env`; the venv is `D:\wildtriage\.venv`; C: has about 1 GB free.
- Every job needs `HF_HOME=D:\wt-env\hf`, `TEMP=D:\wt-env\tmp` and `HF_TOKEN_PATH` pointing at the saved token.
- SAM 3 needs `triton-windows`, and pins numpy 1.26 (so pandas<2.3, pyarrow<20, scipy<1.14).
- Data and results: `D:\wildtriage-data` (embeddings, exp2, depthsam, SA-FARI frames).

## Starting and checking the unattended run
```
powershell -File scripts\lab\start_pipeline.ps1
type D:\wildtriage-data\pipeline_status.json
type D:\wildtriage-data\PIPELINE_REPORT.md          # written at the end
```
Logs: `D:\wt-env\tmp\pipeline_logs\<step>.log` (each also logs the machine's commit level every minute).
It survives closing the terminal, not a logoff or reboot; starting it again resumes where it stopped.

## SAM 3 prompt note
The class-agnostic prompt "animal" often returns no masklets on SA-FARI clips, while the right species noun phrase
finds the animal (score 0.85–0.9). Exp 2 therefore reports MaxN both for "animal" and for the clip's given positive
phrases (VEval protocol, an oracle-prompt upper bound).
