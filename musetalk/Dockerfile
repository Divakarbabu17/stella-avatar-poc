# MuseTalk 1.5 service image - NVIDIA GPU machine only (needs NVIDIA Container Toolkit).
# Follows the official installation steps from https://github.com/TMElyralab/MuseTalk:
#   Python 3.10, PyTorch 2.0.1 + CUDA 11.8 wheels, requirements.txt, MMLab via mim, FFmpeg.
# Model weights are NOT baked in; they live in the ./models volume (download_weights.sh).

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip git curl ca-certificates ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python

# Pinned upstream commit (main as of 2026-11-13). Change deliberately, then re-test.
ARG MUSETALK_REF=0a89dec45a0192b824e3cf4daf96c239440c5ed8
RUN git clone https://github.com/TMElyralab/MuseTalk.git /opt/MuseTalk \
    && git -C /opt/MuseTalk checkout ${MUSETALK_REF}
WORKDIR /opt/MuseTalk

# Official step "Install PyTorch 2.0.1" (Option 1: pip, CUDA 11.8 wheels).
RUN python -m pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Official step "Install Dependencies".
RUN python -m pip install --no-cache-dir -r requirements.txt

# Official step "Install MMLab Packages".
RUN python -m pip install --no-cache-dir -U openmim \
    && mim install mmengine \
    && mim install "mmcv==2.0.1" \
    && mim install "mmdet==3.1.0" \
    && mim install "mmpose==1.1.0"

# HTTP layer. gradio (from requirements.txt) already depends on FastAPI/uvicorn/python-multipart;
# this only guarantees minimum versions without fighting gradio's pins.
RUN python -m pip install --no-cache-dir "fastapi>=0.115" "uvicorn>=0.30" "python-multipart>=0.0.18"

COPY service /opt/stella/service
COPY download_weights.sh /opt/stella/download_weights.sh

ENV MUSETALK_DIR=/opt/MuseTalk \
    MUSETALK_AVATARS_DIR=/data/avatars \
    MUSETALK_OUTPUT_DIR=/data/outputs \
    PYTHONPATH=/opt/stella

EXPOSE 8010
WORKDIR /opt/stella
# Single worker: one copy of the models on the GPU.
CMD ["uvicorn", "service.server:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
