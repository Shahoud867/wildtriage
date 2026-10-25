from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def configure(url: str | None = None) -> None:
    """(Re)create the engine. Tests call this with a SQLite URL."""
    global _engine, _session_factory
    url = url or get_settings().database_url
    if url.startswith("sqlite"):
        kwargs = {"connect_args": {"check_same_thread": False}}
    else:
        kwargs = {"pool_pre_ping": True}
    _engine = create_engine(url, **kwargs)
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)


def get_engine() -> Engine:
    if _engine is None:
        configure()
    return _engine


def session_factory() -> sessionmaker[Session]:
    if _session_factory is None:
        configure()
    return _session_factory


def get_db() -> Iterator[Session]:
    db = session_factory()()
    try:
        yield db
    finally:
        db.close()
