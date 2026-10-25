"""Projects, deployments, datasets and uploads."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from sqlalchemy import select

from app.api.deps import DB, Admin, CurrentUser, Reviewer
from app.core.config import get_settings
from app.models import Dataset, Deployment, MediaItem, Project
from app.models.enums import JobType, Role
from app.schemas import (
    DatasetIn,
    DatasetOut,
    DeploymentIn,
    DeploymentOut,
    ProjectIn,
    ProjectOut,
    UploadOut,
    UploadResult,
)
from app.services import audit
from app.services.ingest import UploadError, media_kind, spool_and_validate
from app.storage import get_storage
from app.workers import queue

router = APIRouter(tags=["projects"])


def _get(db, model, id_: int):
    obj = db.get(model, id_)
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{model.__name__} {id_} not found")
    return obj


@router.get("/projects", response_model=list[ProjectOut])
def list_projects(db: DB, user: CurrentUser):
    return db.execute(select(Project).order_by(Project.id)).scalars().all()


@router.post("/projects", response_model=ProjectOut, status_code=201)
def create_project(body: ProjectIn, db: DB, user: Admin):
    project = Project(**body.model_dump(), created_by=user.id)
    db.add(project)
    db.flush()
    audit.record(db, actor_id=user.id, action="create", entity_type="project", entity_id=project.id,
                 after=body.model_dump())
    db.commit()
    return project


@router.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: int, db: DB, user: CurrentUser):
    return _get(db, Project, project_id)


@router.get("/projects/{project_id}/deployments", response_model=list[DeploymentOut])
def list_deployments(project_id: int, db: DB, user: CurrentUser):
    rows = db.execute(select(Deployment).where(Deployment.project_id == project_id)).scalars().all()
    if user.role != Role.admin:  # endangered species: non-admins only see coarse locations (~1 km)
        for d in rows:
            db.expunge(d)
            if d.latitude is not None:
                d.latitude, d.longitude = round(d.latitude, 2), round(d.longitude, 2)
    return rows


@router.post("/deployments", response_model=DeploymentOut, status_code=201)
def create_deployment(body: DeploymentIn, db: DB, user: Admin):
    _get(db, Project, body.project_id)
    dep = Deployment(**body.model_dump())
    db.add(dep)
    db.commit()
    return dep


@router.get("/projects/{project_id}/datasets", response_model=list[DatasetOut])
def list_datasets(project_id: int, db: DB, user: CurrentUser):
    return db.execute(select(Dataset).where(Dataset.project_id == project_id)).scalars().all()


@router.post("/datasets", response_model=DatasetOut, status_code=201)
def create_dataset(body: DatasetIn, db: DB, user: Admin):
    _get(db, Project, body.project_id)
    ds = Dataset(**body.model_dump(), created_by=user.id)
    db.add(ds)
    db.commit()
    return ds


@router.post("/datasets/{dataset_id}/uploads", response_model=UploadOut)
def upload_media(dataset_id: int, files: list[UploadFile], db: DB, user: Reviewer,
                 deployment_id: int | None = None, auto_infer: bool = True):
    """Validate each file by content, de-duplicate by SHA-256 within the dataset, store it, then enqueue ingest."""
    dataset = _get(db, Dataset, dataset_id)
    storage = get_storage()
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    results, accepted = [], []
    for f in files:
        name = Path(f.filename or "unnamed").name
        try:
            spooled = spool_and_validate(f.file, storage.temp_dir(), max_bytes)
        except UploadError as e:
            results.append(UploadResult(filename=name, ok=False, code=e.code, message=e.message))
            continue
        try:
            kind = media_kind(spooled.sniffed, dataset.modality)
        except UploadError as e:
            spooled.path.unlink(missing_ok=True)
            results.append(UploadResult(filename=name, ok=False, code=e.code, message=e.message))
            continue
        existing = db.execute(
            select(MediaItem.id).where(MediaItem.dataset_id == dataset_id, MediaItem.sha256 == spooled.sha256)
        ).scalar_one_or_none()
        if existing is not None:
            spooled.path.unlink(missing_ok=True)
            results.append(UploadResult(filename=name, ok=False, media_id=existing, code="DUPLICATE",
                                        message="identical file already in this dataset"))
            continue
        key = f"datasets/{dataset_id}/{spooled.sha256[:2]}/{spooled.sha256}{spooled.sniffed.extension}"
        uri = storage.put_file(key, spooled.path)
        media = MediaItem(dataset_id=dataset_id, deployment_id=deployment_id, kind=kind, original_filename=name,
                          storage_uri=uri, sha256=spooled.sha256, size_bytes=spooled.size_bytes)
        db.add(media)
        db.flush()
        accepted.append(media.id)
        results.append(UploadResult(filename=name, ok=True, media_id=media.id))
    job_id = None
    if accepted:
        job = queue.enqueue(db, type=JobType.ingest, queue="cpu", project_id=dataset.project_id,
                            params={"media_ids": accepted, "auto_infer": auto_infer},
                            total_items=len(accepted), created_by=user.id, commit=False)
        job_id = job.id
    db.commit()
    return UploadOut(results=results, job_id=job_id)
