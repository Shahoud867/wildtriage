from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every field can be set as an environment variable with the WT_ prefix."""

    # Reads the repo-root .env when started from backend/ (later files win); real env vars override both.
    model_config = SettingsConfigDict(env_prefix="WT_", env_file=("../.env", ".env"), extra="ignore")

    env: str = "dev"
    database_url: str = "postgresql+psycopg://wildtriage:wildtriage@localhost:5432/wildtriage"
    media_root: Path = Path("./var/media")
    secret_key: str = "change-me"
    session_cookie_name: str = "wt_session"
    session_max_age_s: int = 60 * 60 * 12
    max_upload_mb: int = 2048
    cors_origins: list[str] = ["http://localhost:5173"]

    # "stub" runs deterministic CPU adapters (development, CI); "real" loads the GPU models.
    model_backend: str = "stub"
    # Blocks non-commercial models/datasets when true (see ml/models.yaml and scripts/license_check.py).
    commercial_mode: bool = False

    # "auto" picks cuda, then mps (Apple silicon), then cpu. Set cuda / mps / cpu to pin it.
    device: str = "auto"

    worker_poll_interval_s: float = 1.0
    worker_heartbeat_timeout_s: int = 300
    # Accelerator memory budget per machine: lab PC (RTX 4000 Ada, 20 GB) 18; T1000 (4 GB) 3.5;
    # MacBook M1 Pro (32 GB unified memory) about 12, leaving room for the OS and the browser.
    gpu_vram_budget_gb: float = 18.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
