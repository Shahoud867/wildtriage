from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select, update

from app.api.deps import DB, Admin, CurrentUser
from app.core.config import get_settings
from app.models import EvaluationRun, ModelVersion
from app.schemas import EvaluationRunOut, ModelVersionOut
from app.services import audit

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelVersionOut])
def list_models(db: DB, user: CurrentUser):
    return db.execute(select(ModelVersion).order_by(ModelVersion.task, ModelVersion.id)).scalars().all()


@router.post("/models/{model_id}/activate", response_model=ModelVersionOut)
def activate_model(model_id: int, db: DB, user: Admin):
    mv = db.get(ModelVersion, model_id)
    if mv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")
    if get_settings().commercial_mode and not mv.commercial_ok:
        raise HTTPException(status.HTTP_409_CONFLICT, f"{mv.name} ({mv.license}) is not cleared for commercial use")
    db.execute(update(ModelVersion).where(ModelVersion.task == mv.task).values(is_active=False))
    mv.is_active = True
    audit.record(db, actor_id=user.id, action="model.activate", entity_type="model_version", entity_id=mv.id)
    db.commit()
    db.refresh(mv)
    return mv


@router.get("/evaluations", response_model=list[EvaluationRunOut])
def list_evaluations(db: DB, user: CurrentUser, task: str | None = None):
    q = select(EvaluationRun).order_by(EvaluationRun.id.desc())
    if task:
        q = q.where(EvaluationRun.task == task)
    return db.execute(q).scalars().all()


@router.get("/evaluations/{run_id}", response_model=EvaluationRunOut)
def get_evaluation(run_id: int, db: DB, user: CurrentUser):
    run = db.get(EvaluationRun, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "evaluation run not found")
    return run
