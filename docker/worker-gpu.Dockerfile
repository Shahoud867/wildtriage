# GPU worker image. PyTorch wheels bundle the CUDA runtime; the host only needs an NVIDIA driver
# and the NVIDIA Container Toolkit. Build context: repository root.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

WORKDIR /app
# CUDA 12.8 wheels (same index as the SAM 3 README). Turing (T1000) and Ada (RTX 4000) are both supported [A: verify sm_75 in W1].
RUN uv pip install --system --no-cache torch torchvision --index-url https://download.pytorch.org/whl/cu128
COPY ml/ ml/
COPY backend/ backend/
RUN uv pip install --system --no-cache -e "ml[gpu]" -e backend
# SAM 3 (gated weights: set HF_TOKEN at runtime after requesting access on Hugging Face).
RUN git clone --depth 1 https://github.com/facebookresearch/sam3.git /opt/sam3 && uv pip install --system --no-cache -e /opt/sam3

WORKDIR /app/backend
CMD ["python", "-m", "app.workers.runner", "--queues", "gpu,gpu-heavy"]
