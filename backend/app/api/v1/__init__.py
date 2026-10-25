from fastapi import APIRouter

from app.api.v1 import analytics, auth, jobs, media, models, projects, review

router = APIRouter(prefix="/api/v1")
for module in (auth, projects, media, jobs, review, models, analytics):
    router.include_router(module.router)
