# WildTriage

A self-hosted camera-trap **triage and verification** platform. It turns raw camera-trap images and videos into verified species records. The main claim is measured: *how much human review it saves at a controlled error rate*.

- **Course project** (Computer Vision, Semester 7): 2 people, 8 weeks (19 Oct – 13 Dec 2026).
- **Full plan:** [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md). It covers paper validation, scope, architecture, the roadmap, evaluation and risks.

| Paper (2026) | Role here |
|---|---|
| SA-FARI (CVPR 2026) | Video detection, segmentation and tracking with SAM 3; the SA-FARI test set for evaluation |
| Beyond Appearance / DepthSAM (CVPR 2026) | Camouflage refinement expert (P1) |
| Lessons and Open Questions… / StreamTrap (ECCV 2026) | Per-site species classification over time; its recipe drives our research contribution |

**Our contribution:** review-budgeted, risk-controlled triage with site adaptation (`ml/wildtriage_ml/triage`, `research/experiments/exp1_triage.py`).

## Repository layout

```
backend/    FastAPI app, SQLAlchemy models, Alembic migrations, Postgres job queue + workers, tests
ml/         wildtriage_ml: model adapters (MegaDetector, BioCLIP 2, SAM 3, stubs), VRAM-aware registry,
            pipeline utils (quality, counting, sampling), triage (calibration, conformal, selection, policy),
            adaptation (Balanced Softmax linear heads, WiSE), tests; models.yaml = license manifest
research/   experiments (Exp 1–4), configs, data prep, smoke tests
frontend/   React 19 + Vite + TypeScript + Tailwind: login, projects, data/upload, jobs, media, review, analytics, models
docker/     Dockerfiles + nginx; docker-compose.yml at the root
scripts/    native/ (no-admin Windows run scripts), license_check.py, seed_demo.py
docs/       PROJECT_PLAN.md, ADRs, dataset/model card templates
third_party/ pinned upstream clones (gitignored)
```

## Machines

| Machine | Use | `.env` |
|---|---|---|
| Lab PC: RTX 4000 Ada 20 GB, i7-14700, 32 GB (managed Windows) | Heavy GPU work (SAM 3, DepthSAM), all raw datasets, demo host | `WT_MODEL_BACKEND=real`, `WT_GPU_VRAM_BUDGET_GB=18`; worker queues `gpu,gpu-heavy` |
| Teammate's PC: Quadro T1000 4 GB, i7 9th gen, 32 GB | Light GPU (MegaDetector, BioCLIP 2 in fp16), data prep | `WT_MODEL_BACKEND=real`, `WT_GPU_VRAM_BUDGET_GB=3.5`; worker queue `gpu` |
| MacBook Pro M1 Pro, 32 GB unified memory, 512 GB | Light models on Apple's MPS backend (fp32), development, Exp 1 on cached embeddings. **No SAM 3 or DepthSAM** (CUDA only): those stay on the lab PC, or molab when needed | `WT_MODEL_BACKEND=real`, `WT_DEVICE=mps`, `WT_GPU_VRAM_BUDGET_GB=12`; worker queue `gpu` |
| Laptop: i5-1235U, 20 GB, no CUDA | Platform development with stub models; Exp 1 on cached embeddings | `WT_MODEL_BACKEND=stub` |

Device selection is automatic: `WT_DEVICE=auto` picks CUDA, then MPS, then CPU (`ml/wildtriage_ml/device.py`).
- Precision follows the device: bf16 on the lab PC, fp16 on the T1000, fp32 on the Mac and CPU.
- A model that cannot run on the worker's device (SAM 3 on a Mac) is refused with a message to route it to the `gpu-heavy` queue.

> **Move this folder out of OneDrive before installing anything.** Syncing `.venv`, `node_modules` and data causes locks. Use e.g. `C:\dev\wildtriage` (Windows) or `~/dev/wildtriage` (Mac). Then `git init`.

## Quick start

**Laptop with Docker (stub models):**

```bash
cp .env.example .env
docker compose --profile stub up --build
docker compose exec api python -m app.cli create-user --email you@example.com --name You --role admin
```

The app is at http://localhost:8080 and the API docs at http://localhost:8000/docs.

**Native Windows, no admin rights (lab PC or any machine):**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\native\setup.ps1 -Machine lab   # or t1000 / laptop
powershell -ExecutionPolicy Bypass -File scripts\native\start-postgres.ps1        # portable PostgreSQL zip
powershell -ExecutionPolicy Bypass -File scripts\native\start-api.ps1
powershell -ExecutionPolicy Bypass -File scripts\native\start-worker.ps1 -Queues cpu
powershell -ExecutionPolicy Bypass -File scripts\native\start-worker.ps1 -Queues "gpu,gpu-heavy"
powershell -ExecutionPolicy Bypass -File scripts\native\start-frontend.ps1        # http://localhost:5173
```

**Native macOS or Linux** (MacBook, or a Linux machine):

```bash
bash scripts/native/setup.sh mac          # or: cuda (Linux + NVIDIA) / cpu (stub models)
bash scripts/native/start-postgres.sh     # Homebrew postgresql@16 or Postgres.app
bash scripts/native/start-api.sh
bash scripts/native/start-worker.sh cpu
bash scripts/native/start-worker.sh gpu   # on the Mac; gpu-heavy is refused there (needs CUDA)
bash scripts/native/start-frontend.sh     # http://localhost:5173
```

Docker also works on the Mac for Postgres, the API and the frontend. Containers on a Mac get no GPU, so run model workers natively there.

## Tests

```bash
cd ml && python -m pytest -q            # triage math, registry, policy, pipeline utils
cd backend && python -m pytest -q       # API, uploads, queue, migrations, end-to-end stub pipeline
cd research && python -m pytest -q      # Exp 1 simulator smoke tests
cd frontend && npm ci && npm run build && npm test
python research/experiments/exp1_triage.py --synthetic   # policy comparison on synthetic streams
python scripts/license_check.py                          # NC / unverified licenses
python -m pytest -q scripts/tests                        # Windows and macOS/Linux scripts match; bash syntax
```

## Skeleton status (18 Oct 2026)

**Works and is tested:**
- upload with content sniffing, SHA-256 dedup and per-file error codes;
- the Postgres job queue (priorities, retries, stale-job requeue, cancellation);
- ingest → inference (stub models) → calibrated triage → review queue → audited decisions → analytics;
- role-based auth;
- the Alembic migration (SQLite, plus Postgres DDL rendered offline);
- triage math (temperature scaling, ECE, CRC, LTT, ACI, selection strategies);
- linear heads with Balanced Softmax + WiSE;
- the Exp 1 simulator;
- the frontend build.

**Written but not yet run against real models or services:**
- the MegaDetector and BioCLIP 2 adapters (the MegaDetector API calls are marked `[A] verify`);
- Docker images;
- native PowerShell scripts;
- Postgres at runtime.

**Stubs with week markers:**
- SAM 3 adapter (W3);
- DepthSAM / OWL (P1);
- LoRA (W5);
- Exp 2–4;
- exports;
- settings UI.
