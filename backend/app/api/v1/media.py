from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select

from app.api.deps import DB, CurrentUser
from app.models import Classification, Detection, MediaItem
from app.models.enums import MediaKind, MediaStatus
from app.schemas import DetectionOut, MediaOut
from app.storage import get_storage

router = APIRouter(prefix="/media", tags=["media"])

MIME = {".jpg": "image/jpeg", ".png": "image/png", ".tif": "image/tiff", ".mp4": "video/mp4",
        ".mov": "video/quicktime", ".avi": "video/x-msvideo"}


def _media(db, media_id: int) -> MediaItem:
    media = db.get(MediaItem, media_id)
    if media is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "media not found")
    return media


@router.get("", response_model=list[MediaOut])
def list_media(db: DB, user: CurrentUser, dataset_id: int | None = None, status_: MediaStatus | None = Query(None, alias="status"),
               kind: MediaKind | None = None, is_night: bool | None = None, limit: int = 100, offset: int = 0):
    q = select(MediaItem).order_by(MediaItem.id)
    if dataset_id is not None:
        q = q.where(MediaItem.dataset_id == dataset_id)
    if status_ is not None:
        q = q.where(MediaItem.status == status_)
    if kind is not None:
        q = q.where(MediaItem.kind == kind)
    if is_night is not None:
        q = q.where(MediaItem.is_night == is_night)
    return db.execute(q.limit(min(limit, 500)).offset(offset)).scalars().all()


@router.get("/{media_id}", response_model=MediaOut)
def get_media(media_id: int, db: DB, user: CurrentUser):
    return _media(db, media_id)


@router.get("/{media_id}/file")
def get_media_file(media_id: int, db: DB, user: CurrentUser):
    # P1: signed URLs from an S3-compatible store instead of streaming through the API.
    media = _media(db, media_id)
    path = get_storage().local_path(media.storage_uri)
    return FileResponse(path, media_type=MIME.get(path.suffix, "application/octet-stream"))


@router.get("/{media_id}/predictions", response_model=list[DetectionOut])
def get_predictions(media_id: int, db: DB, user: CurrentUser):
    _media(db, media_id)
    rows = db.execute(
        select(Detection, Classification)
        .outerjoin(Classification, Classification.detection_id == Detection.id)
        .where(Detection.media_id == media_id)
        .order_by(Detection.id)
    ).all()
    out = []
    for det, cls in rows:
        item = DetectionOut.model_validate(det)
        if cls is not None:
            item.label, item.calibrated_score, item.topk = cls.label, cls.calibrated_score, cls.topk
        out.append(item)
    return out
