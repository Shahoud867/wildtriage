import os
import struct
import tempfile
import zlib
from pathlib import Path

# Configure before importing the app: a throwaway SQLite database, media folder and stub models.
_TMP = Path(tempfile.mkdtemp(prefix="wt-test-"))
os.environ["WT_DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["WT_MEDIA_ROOT"] = str(_TMP / "media")
os.environ["WT_MODEL_BACKEND"] = "stub"
os.environ["WT_ENV"] = "dev"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.models  # noqa: E402,F401
from app.core.security import hash_password  # noqa: E402
from app.db import session as db_session  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.models import User  # noqa: E402
from app.models.enums import Role  # noqa: E402

db_session.configure(os.environ["WT_DATABASE_URL"])
PASSWORD = "pw-123456"


@pytest.fixture(autouse=True)
def fresh_db():
    engine = db_session.get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def db():
    with db_session.session_factory()() as s:
        yield s


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def make_user(role: str) -> int:
    with db_session.session_factory()() as s:
        user = User(email=f"{role}@test.local", name=role, role=Role(role), password_hash=hash_password(PASSWORD))
        s.add(user)
        s.commit()
        return user.id


def login(client: TestClient, role: str) -> TestClient:
    make_user(role)
    r = client.post("/api/v1/auth/login", json={"email": f"{role}@test.local", "password": PASSWORD})
    assert r.status_code == 200, r.text
    return client


@pytest.fixture
def admin_client(client):
    return login(client, "admin")


def make_png(w: int = 16, h: int = 16, rgb: tuple[int, int, int] = (10, 120, 30)) -> bytes:
    """A valid PNG without needing Pillow."""
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def create_project_and_dataset(client: TestClient, modality: str = "camera_trap") -> tuple[int, int]:
    p = client.post("/api/v1/projects", json={"name": "Osa demo"}).json()
    d = client.post("/api/v1/datasets", json={"project_id": p["id"], "name": "card-01", "modality": modality}).json()
    return p["id"], d["id"]
