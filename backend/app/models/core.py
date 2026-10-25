"""Users, projects, taxonomy and deployments."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, BigIntKey, JSONType, TimestampMixin, enum_col, fk, pk
from app.models.enums import Modality, Role


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = pk()
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[Role] = mapped_column(enum_col(Role), default=Role.viewer)
    is_active: Mapped[bool] = mapped_column(default=True)


class Project(TimestampMixin, Base):
    __tablename__ = "projects"

    id: Mapped[int] = pk()
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    # alpha (target auto-accept error), review_budget, country (ISO 3166-1 alpha-3), location_precision_m, ...
    settings: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_by: Mapped[int | None] = fk("users.id", nullable=True, ondelete="SET NULL")


class Taxon(Base):
    __tablename__ = "taxa"

    id: Mapped[int] = pk()
    scientific_name: Mapped[str] = mapped_column(String(255), unique=True)
    common_name: Mapped[str] = mapped_column(String(255), default="")
    rank: Mapped[str] = mapped_column(String(32), default="species")
    parent_id: Mapped[int | None] = fk("taxa.id", nullable=True, ondelete="SET NULL")
    gbif_key: Mapped[int | None] = mapped_column(nullable=True)


class ProjectSpecies(Base):
    __tablename__ = "project_species"

    project_id: Mapped[int] = mapped_column(BigIntKey, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True)
    taxon_id: Mapped[int] = mapped_column(BigIntKey, ForeignKey("taxa.id", ondelete="CASCADE"), primary_key=True)
    # Optional text prompt override for open-vocabulary models (BioCLIP 2, SAM 3).
    prompt: Mapped[str | None] = mapped_column(String(512), nullable=True)


class Deployment(TimestampMixin, Base):
    """A camera (or survey flight) at a site for a period of time."""

    __tablename__ = "deployments"

    id: Mapped[int] = pk()
    project_id: Mapped[int] = fk("projects.id")
    name: Mapped[str] = mapped_column(String(255))
    modality: Mapped[Modality] = mapped_column(enum_col(Modality), default=Modality.camera_trap)
    camera_model: Mapped[str] = mapped_column(String(255), default="")
    # Precise coordinates are only returned to admins; others get rounded values (endangered species).
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
