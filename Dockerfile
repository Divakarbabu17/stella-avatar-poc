# syntax=docker/dockerfile:1
# One Dockerfile, one build target per service (see compose.yaml):
#
#   orchestrator   python-slim + FastAPI/httpx (no ML packages)
#   whisper        torch-base + openai-whisper
#   qwen           torch-base + transformers
#   kokoro         torch-base + transformers + kokoro/spaCy
#   cli            all three, for the standalone debugging CLIs
#
# Package layers come before `COPY app`, so code edits rebuild in seconds.
# The pip cache mount keeps downloaded wheels between builds.

# ---------- orchestrator ----------
FROM python:3.11-slim AS orchestrator

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements/api.txt requirements/orchestrator.txt requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements/api.txt -r requirements/orchestrator.txt

COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- shared ML base: FFmpeg + CPU-only PyTorch ----------
FROM python:3.11-slim AS torch-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# CPU wheel index: no CUDA libraries. Installed first so no later
# requirement can pull a CUDA build of torch.
RUN python -m pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements/api.txt requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements/api.txt

# ---------- whisper ----------
FROM torch-base AS whisper-deps
COPY requirements/whisper.txt requirements/whisper-nodeps.txt requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements/whisper.txt \
    && python -m pip install --no-deps -r requirements/whisper-nodeps.txt

FROM whisper-deps AS whisper
COPY app ./app
EXPOSE 8001
CMD ["uvicorn", "app.servers.whisper:app", "--host", "0.0.0.0", "--port", "8001"]

# ---------- qwen ----------
FROM torch-base AS qwen-deps
COPY requirements/qwen.txt requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements/qwen.txt

FROM qwen-deps AS qwen
COPY app ./app
EXPOSE 8002
CMD ["uvicorn", "app.servers.qwen:app", "--host", "0.0.0.0", "--port", "8002"]

# ---------- kokoro (reuses the qwen-deps layers: same transformers) ----------
FROM qwen-deps AS kokoro-deps
COPY requirements/kokoro.txt requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements/kokoro.txt

FROM kokoro-deps AS kokoro
COPY app ./app
EXPOSE 8003
CMD ["uvicorn", "app.servers.kokoro:app", "--host", "0.0.0.0", "--port", "8003"]

# ---------- cli: every model in one image, for the debugging CLIs ----------
FROM kokoro-deps AS cli
COPY requirements/whisper.txt requirements/whisper-nodeps.txt requirements/orchestrator.txt requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements/whisper.txt -r requirements/orchestrator.txt \
    && python -m pip install --no-deps -r requirements/whisper-nodeps.txt
COPY app ./app
CMD ["python", "-m", "app.main", "--help"]
