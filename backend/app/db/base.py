from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import JSON, BigInteger, DateTime, Enum, ForeignKey, Integer, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# BIGINT keys on Postgres; SQLite (tests) only auto-increments INTEGER PRIMARY KEY.
BigIntKey = BigInteger().with_variant(Integer(), "sqlite")
# JSONB on Postgres (indexable), plain JSON elsewhere.
JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


def pk() -> Mapped[int]:
    return mapped_column(BigIntKey, primary_key=True, autoincrement=True)


def fk(target: str, *, nullable: bool = False, ondelete: str = "CASCADE", index: bool = True) -> Mapped:
    return mapped_column(BigIntKey, ForeignKey(target, ondelete=ondelete), nullable=nullable, index=index)


def enum_col(enum_cls: type[StrEnum]) -> Enum:
    """Portable enum: VARCHAR + CHECK constraint, stores the enum value."""
    return Enum(
        enum_cls,
        native_enum=False,
        length=32,
        validate_strings=True,
        values_callable=lambda e: [m.value for m in e],
    )
