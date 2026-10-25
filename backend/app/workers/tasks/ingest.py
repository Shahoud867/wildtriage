"""Ingest: metadata and quality features, then hand off to inference on the right GPU queue."""

from sqlalchemy.orm import Session

from app.models import Job, MediaItem
from app.models.enums import JobType, MediaKind, MediaStatus
from app.storage import get_storage
from app.workers import queue
from app.workers.tasks import task


def _image_quality(path) -> tuple[dict, int | None, int | None]:
    try:
        import numpy as np
        from PIL import Image
        from wildtriage_ml.pipeline.quality import quality_features
    except ImportError:
        return {}, None, None  # Pillow not installed (e.g. minimal CI); quality is filled later
    with Image.open(path) as img:
        img.thumbnail((512, 512))  # quality features do not need full resolution
        arr = np.asarray(img.convert("RGB"))
        width, height = Image.open(path).size
    return quality_features(arr), width, height


@task(JobType.ingest)
def run_ingest(db: Session, job: Job) -> dict:
    storage = get_storage()
    ids = job.params.get("media_ids", [])
    images, videos = [], []
    for i, media_id in enumerate(ids):
        media = db.get(MediaItem, media_id)
        if media is None:
            continue
        if media.kind == MediaKind.video:
            # W3: ffprobe for duration/fps/size; ffmpeg frame extraction at 1 fps (screening) and 6 fps (SAM 3).
            videos.append(media.id)
        else:
            quality, width, height = _image_quality(storage.local_path(media.storage_uri))
            media.quality, media.width, media.height = quality, width, height
            media.is_night = quality.get("is_night")
            images.append(media.id)
        media.status = MediaStatus.validated
        queue.heartbeat(db, job, done_items=i + 1)

    common = {"species": job.params.get("species", [])}
    if job.params.get("auto_infer", True):
        # Images go to "gpu" (any CUDA machine, incl. the T1000); video needs SAM 3 -> "gpu-heavy" (lab PC).
        if images:
            queue.enqueue(db, type=JobType.inference, queue="gpu", project_id=job.project_id,
                          params={**common, "media_ids": images}, total_items=len(images))
        if videos:
            queue.enqueue(db, type=JobType.inference, queue="gpu-heavy", project_id=job.project_id,
                          params={**common, "media_ids": videos}, total_items=len(videos))
    db.commit()
    return {"validated": len(images) + len(videos), "images": len(images), "videos": len(videos)}
