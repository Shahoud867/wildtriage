# API and CPU worker image (no CUDA). Build context: repository root.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY ml/ ml/
COPY backend/ backend/
RUN uv pip install --system --no-cache -e ml -e backend pillow

WORKDIR /app/backend
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
