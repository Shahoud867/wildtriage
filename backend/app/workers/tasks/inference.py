"""Inference: detection -> species -> calibration -> triage, persisted with model versions."""

from functools import lru_cache

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import (
    Classification,
    Detection,
    Job,
    MediaItem,
    ModelVersion,
    ProjectSpecies,
    ReviewTask,
    Taxon,
    Track,
    TriageDecision,
)
from app.models.enums import DetectionCategory, JobType, MediaKind, MediaStatus, ModelTask, TriageOutcome
from app.storage import get_storage
from app.workers import queue
from app.workers.tasks import task

# Used only when a project has no species list yet.
FALLBACK_SPECIES = ["Panthera onca", "Tapirus bairdii", "Pecari tajacu", "Puma concolor", "Dasyprocta punctata"]


@lru_cache
def get_pipeline():
    from wildtriage_ml.runtime import Pipeline

    s = get_settings()
    return Pipeline(backend=s.model_backend, vram_budget_gb=s.gpu_vram_budget_gb, device=s.device)


def _model_version(db: Session, info, task_: ModelTask, config: dict | None = None) -> ModelVersion:
    mv = db.execute(
        select(ModelVersion).where(ModelVersion.name == info.name, ModelVersion.version == info.version)
    ).scalar_one_or_none()
    if mv is None:
        mv = ModelVersion(name=info.name, version=info.version, task=task_, framework=info.framework,
                          license=info.license, commercial_ok=info.commercial_ok, config=config or {},
                          is_active=True)
        db.add(mv)
        db.commit()  # committed on its own so a failed media item cannot roll it back
    return mv


def _species(db: Session, job: Job) -> list[str]:
    if job.params.get("species"):
        return list(job.params["species"])
    if job.project_id is not None:
        names = db.execute(
            select(Taxon.scientific_name).join(ProjectSpecies, ProjectSpecies.taxon_id == Taxon.id)
            .where(ProjectSpecies.project_id == job.project_id)
        ).scalars().all()
        if names:
            return list(names)
    return FALLBACK_SPECIES


def _triage(db, job, media, policy_mv, decision, detection_id=None):
    td = TriageDecision(media_id=media.id, detection_id=detection_id, policy_version_id=policy_mv.id,
                        decision=TriageOutcome(decision.outcome), risk=decision.risk, reasons=decision.reasons)
    db.add(td)
    db.flush()
    if decision.outcome == "review" and job.project_id is not None:
        db.add(ReviewTask(project_id=job.project_id, triage_decision_id=td.id, media_id=media.id,
                          detection_id=detection_id, priority=decision.priority))
    return td


def _process_image(db, job, media, pipe, species, mvs, path):
    from wildtriage_ml.triage.calibration import softmax

    cfg = pipe.policy.config
    all_dets = pipe.detector.detect(path)
    dets = [d for d in all_dets if d.score >= cfg.detection_min_score]
    animals = [d for d in dets if d.category == "animal"]
    for idx, d in enumerate(dets):
        det = Detection(media_id=media.id, model_version_id=mvs["detect"].id, job_id=job.id,
                        category=DetectionCategory(d.category), x=d.x, y=d.y, w=d.w, h=d.h, score=d.score)
        db.add(det)
        db.flush()
        if d.category != "animal":
            continue  # people/vehicles: flagged for privacy handling, not classified
        scores = pipe.classifier.classify(path, d, species)
        probs = pipe.calibrator.transform(scores.logits[None, :])[0]
        order = probs.argsort()[::-1]
        top1 = float(probs[order[0]])
        top2 = float(probs[order[1]]) if len(order) > 1 else 0.0
        raw = softmax(scores.logits[None, :])[0]
        db.add(Classification(detection_id=det.id, model_version_id=mvs["classify"].id,
                              topk=[{"label": scores.labels[j], "p": float(probs[j])} for j in order[:5]],
                              label=scores.labels[order[0]], raw_score=float(raw[order[0]]), calibrated_score=top1))
        decision = pipe.policy.decide_detection(top1, top2, d.area, media.quality or {}, f"{media.id}:{idx}")
        _triage(db, job, media, mvs["policy"], decision, detection_id=det.id)
    if not animals:
        max_score = max((d.score for d in all_dets if d.category == "animal"), default=0.0)
        _triage(db, job, media, mvs["policy"], pipe.policy.decide_empty(max_score, media.quality or {}))


def _process_video(db, job, media, pipe, species, mvs, path):
    from wildtriage_ml.pipeline.counting import max_n

    tracks = pipe.video_tracker.track(path, ["animal", *species])
    for t in tracks:
        db.add(Track(media_id=media.id, model_version_id=mvs["segment_track"].id, job_id=job.id,
                     track_key=t.track_key, start_frame=min(t.frames), end_frame=max(t.frames),
                     label=t.label, score=t.score))
    media.quality = {**(media.quality or {}), "max_n": max_n(tracks)}
    # W3: per-track species vote + triage per track; until then every clip with animals is reviewed.
    if tracks:
        from wildtriage_ml.triage.policy import Decision

        _triage(db, job, media, mvs["policy"], Decision("review", 0.5, ["video_track_review"]))
    else:
        _triage(db, job, media, mvs["policy"], pipe.policy.decide_empty(0.0, media.quality or {}))


@task(JobType.inference)
def run_inference(db: Session, job: Job) -> dict:
    pipe = get_pipeline()
    storage = get_storage()
    species = _species(db, job)
    ids = job.params.get("media_ids", [])
    from wildtriage_ml.adapters.base import ModelInfo

    policy_info = ModelInfo("triage-policy", "default", "triage", "project", True, vram_gb=0.0, framework="numpy")
    mvs = {"policy": _model_version(db, policy_info, ModelTask.triage, pipe.policy.config.to_dict())}
    processed = 0
    for i, media_id in enumerate(ids):
        if queue.is_cancelled(db, job):
            return {"processed": processed, "cancelled": True}
        media = db.get(MediaItem, media_id)
        if media is None:
            continue
        media.status = MediaStatus.processing
        path = storage.local_path(media.storage_uri)
        try:
            if media.kind == MediaKind.video:
                mvs.setdefault("segment_track", _model_version(db, pipe.video_tracker.info, ModelTask.segment_track))
                _process_video(db, job, media, pipe, species, mvs, path)
            else:
                mvs.setdefault("detect", _model_version(db, pipe.detector.info, ModelTask.detect))
                mvs.setdefault("classify", _model_version(db, pipe.classifier.info, ModelTask.classify))
                _process_image(db, job, media, pipe, species, mvs, path)
            media.status = MediaStatus.processed
            processed += 1
        except Exception as exc:  # one bad file must not fail the whole batch
            db.rollback()
            media = db.get(MediaItem, media_id)
            media.status, media.error = MediaStatus.failed, f"{type(exc).__name__}: {exc}"
        queue.heartbeat(db, job, done_items=i + 1)
    return {"processed": processed, "failed": len(ids) - processed}
