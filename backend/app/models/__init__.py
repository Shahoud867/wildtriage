"""Import every model so Base.metadata is complete (Alembic, create_all)."""

from app.models.core import Deployment, Project, ProjectSpecies, Taxon, User
from app.models.data import Dataset, DatasetVersion, Frame, MediaItem
from app.models.evaluation import Annotation, AuditLog, EvaluationRun, Export
from app.models.ml import Job, ModelVersion
from app.models.predictions import Classification, Detection, Track
from app.models.review import ReviewTask, TriageDecision, VerificationDecision

__all__ = [
    "Annotation",
    "AuditLog",
    "Classification",
    "Dataset",
    "DatasetVersion",
    "Deployment",
    "Detection",
    "EvaluationRun",
    "Export",
    "Frame",
    "Job",
    "MediaItem",
    "ModelVersion",
    "Project",
    "ProjectSpecies",
    "ReviewTask",
    "Taxon",
    "Track",
    "TriageDecision",
    "User",
    "VerificationDecision",
]
