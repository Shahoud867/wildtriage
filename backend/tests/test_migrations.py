import tempfile
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from app.core.config import get_settings
from app.db.base import Base

BACKEND = Path(__file__).resolve().parents[1]


def test_migrations_match_models_and_downgrade_cleanly(monkeypatch):
    db_path = Path(tempfile.mkdtemp(prefix="wt-mig-")) / "m.db"
    url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("WT_DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        cfg = Config(str(BACKEND / "alembic.ini"))
        cfg.set_main_option("script_location", str(BACKEND / "alembic"))
        command.upgrade(cfg, "head")
        tables = set(inspect(create_engine(url)).get_table_names()) - {"alembic_version"}
        assert tables == set(Base.metadata.tables), "run `alembic revision --autogenerate` after model changes"
        command.downgrade(cfg, "base")
        assert set(inspect(create_engine(url)).get_table_names()) <= {"alembic_version"}
    finally:
        get_settings.cache_clear()  # monkeypatch restores the env var; drop the cached settings too
