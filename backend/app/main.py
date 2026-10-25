from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.v1 import router as v1_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import session_factory


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.env)
    app = FastAPI(title="WildTriage API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(v1_router)

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        try:
            with session_factory()() as db:
                db.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            db_ok = False
        return {"status": "ok" if db_ok else "degraded", "db": db_ok, "model_backend": settings.model_backend}

    return app


app = create_app()
