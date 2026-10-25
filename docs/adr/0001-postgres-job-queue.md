# ADR 0001: Postgres-backed job queue instead of Celery + Redis

**Status:** accepted (18 Oct 2026)

## Context

- The main GPU machine is a university-managed Windows 11 Enterprise PC (RTX 4000 Ada, 20 GB). Admin rights, Docker and WSL2 there are unconfirmed.
- Redis has no official Windows build, and Celery's Windows support is limited.
- Job volume is low: one or two GPU workers, thousands of jobs.
- Workers run on different machines with different capabilities:
  - the lab PC: 20 GB, all models;
  - the teammate's PC: Quadro T1000 4 GB, light models;
  - the laptop: no CUDA, stub models.

## Decision

- Jobs are rows in the `jobs` table. Workers claim the next runnable job with `SELECT … FOR UPDATE SKIP LOCKED` (`backend/app/workers/queue.py`).
- Queues route by capability:
  - `cpu`: ingest and exports;
  - `gpu`: MegaDetector and BioCLIP 2, any CUDA machine;
  - `gpu-heavy`: SAM 3 and DepthSAM, lab PC only.
- Retries use exponential backoff. Heartbeats let stale jobs be requeued after a crash. Running jobs check for cancellation between items.

## Consequences

- One fewer service. "Create job" and "enqueue" are one transaction.
- The same code runs natively on Windows and in Docker.
- No Flower dashboard; the app's Jobs page shows the queue instead.
- If volume grows, swap the implementation behind `enqueue`/`claim_next` for Procrastinate (also Postgres-based) or Celery. The task registry and job schema stay the same.
