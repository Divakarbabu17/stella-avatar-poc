#!/bin/bash
# Download MuseTalk weights into ./models (mounted at /opt/MuseTalk/models in the container).
#
# Same files and target paths as the official MuseTalk download_weights.sh
# (https://github.com/TMElyralab/MuseTalk/blob/main/download_weights.sh). Differences:
#   * HF_ENDPOINT is configurable. The official script hard-codes https://hf-mirror.com;
#     here the default is huggingface.co. Use HF_ENDPOINT=https://hf-mirror.com if needed.
#   * Already-downloaded files are skipped, and the script stops on the first error.
#
# Run on the GPU host, from the project root (writes to musetalk/models/):
#   docker compose run --rm musetalk bash /opt/stella/download_weights.sh
set -euo pipefail

cd "${MUSETALK_DIR:-/opt/MuseTalk}"
CheckpointsDir="models"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper

# huggingface-cli and gdown come from MuseTalk's requirements.txt (huggingface_hub, gdown).

# MuseTalk V1.0 weights (kept for parity with the official layout)
huggingface-cli download TMElyralab/MuseTalk --local-dir $CheckpointsDir \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

# MuseTalk V1.5 weights (unet.pth) - used by this service
huggingface-cli download TMElyralab/MuseTalk --local-dir $CheckpointsDir \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

# SD VAE weights
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir $CheckpointsDir/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"

# Whisper (tiny) weights - MuseTalk's audio encoder
huggingface-cli download openai/whisper-tiny --local-dir $CheckpointsDir/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# DWPose weights
huggingface-cli download yzd-v/DWPose --local-dir $CheckpointsDir/dwpose \
  --include "dw-ll_ucoco_384.pth"

# SyncNet weights
huggingface-cli download ByteDance/LatentSync --local-dir $CheckpointsDir/syncnet \
  --include "latentsync_syncnet.pt"

# Face parsing (BiSeNet) weights from Google Drive, plus the ResNet18 backbone
if [ ! -s $CheckpointsDir/face-parse-bisent/79999_iter.pth ]; then
  # URL form: works with old and new gdown (gdown 5+ removed the --id option).
  gdown "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812" -O $CheckpointsDir/face-parse-bisent/79999_iter.pth
fi
if [ ! -s $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth ]; then
  curl -fL https://download.pytorch.org/models/resnet18-5c106cde.pth \
    -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth
fi

# S3FD face detector. MuseTalk downloads it itself via torch.hub when the service starts
# (musetalk/utils/face_detection/detection/sfd/sfd_detector.py). Pre-fetching it into
# models/torch (TORCH_HOME in compose.gpu.yaml) keeps it persistent and offline-safe.
mkdir -p $CheckpointsDir/torch/hub/checkpoints
if [ ! -s $CheckpointsDir/torch/hub/checkpoints/s3fd-619a316812.pth ]; then
  curl -fL https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth \
    -o $CheckpointsDir/torch/hub/checkpoints/s3fd-619a316812.pth
fi

echo "All MuseTalk weights are in $(pwd)/$CheckpointsDir"
