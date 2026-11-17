# MuseTalk service (`musetalk` in compose.yaml)

This folder holds the image and code for the fifth service of the Stella stack.
It is defined in the project's main [compose.yaml](../compose.yaml) under the
Compose profile `gpu` and runs **only on a host with an NVIDIA GPU**. On the
CPU-only development laptop it is never built or started (see "CPU-only
machines" below).

The service wraps the official
[MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk) **real-time inference**
code. Each avatar is prepared once (face detection, masks, VAE latents), then
every request only runs the UNet and VAE decoder. It receives the Kokoro WAV
from the orchestrator over the internal Compose network and returns a
lip-synced MP4 with that same audio muxed in.

- Upstream: `TMElyralab/MuseTalk` pinned to commit `0a89dec` (`MUSETALK_REF` in the Dockerfile)
- Install steps follow the official README: Python 3.10, `torch==2.0.1` with
  CUDA 11.8 wheels, `requirements.txt`, `mmengine` / `mmcv==2.0.1` /
  `mmdet==3.1.0` / `mmpose==1.1.0` via `mim`, FFmpeg
- Inference settings follow `inference.sh v1.5 realtime`: `--version v15`,
  fp16, 25 fps, `bbox_shift=0`, `parsing_mode=jaw`, `extra_margin=10`, batch size 20

```text
musetalk/
├── Dockerfile            CUDA 11.8 image: official MuseTalk + this service
├── download_weights.sh   official weights → musetalk/models/
├── service/
│   ├── engine.py         loads the models once, prepares avatars, renders MP4s
│   └── server.py         FastAPI: /health /avatars /generate /videos
├── models/               weights (created by download_weights.sh, git-ignored)
└── outputs/              rendered MP4s until the orchestrator fetches them (git-ignored)
```

## Requirements (GPU host)

| | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA, 4 GB VRAM (MuseTalk's own minimum test: RTX 3050 Ti Laptop, slow) | 8 GB+ VRAM: RTX 3060 12 GB, RTX 4070, T4/L4/A10 |
| GPU generation | Turing (RTX 20xx) or newer, up to Ada (RTX 40xx) / Hopper | |
| Driver | NVIDIA driver with CUDA 11.8 support (Linux ≥ 520.61) | latest stable |
| OS | Ubuntu 22.04 (or Windows 11 + WSL2 + Docker Desktop GPU support) | Ubuntu 22.04 |
| Software | Docker Engine + Compose v2, NVIDIA Container Toolkit | |
| Disk | ~35 GB (MuseTalk image ~15 GB, weights ~10 GB, voice images + models) | 50 GB |
| RAM | 16 GB (MuseTalk + Whisper/Qwen3/Kokoro on CPU) | 32 GB |

> **RTX 50-series (Blackwell) is not supported** by the official PyTorch
> 2.0.1 + CUDA 11.8 stack MuseTalk pins.

Official performance claim: 30 fps+ on a Tesla V100 in real-time mode.

Whisper, Qwen3 and Kokoro keep running on the host's CPU, exactly as on the
laptop; only MuseTalk uses the GPU.

## Run the full stack on the GPU host

All commands run from the **project root** (the folder with `compose.yaml`).

1. Prepare Docker for the GPU (once):

   ```bash
   nvidia-smi                                   # driver works
   # Install NVIDIA Container Toolkit:
   #   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

2. Configure: `cp .env.example .env`, then in `.env`:

   ```ini
   COMPOSE_PROFILES=gpu      # include the musetalk service in every compose command
   MUSETALK_ENABLED=true     # orchestrator renders avatar videos
   ```

3. Build, download the weights, add Stella's reference:

   ```bash
   docker compose build                                              # MuseTalk image: 20-40 min the first time
   docker compose run --rm musetalk bash /opt/stella/download_weights.sh
   cp /path/to/stella.mp4 avatars/stella/reference.mp4               # see avatars/stella/README.md
   ```

4. Start and check:

   ```bash
   docker compose up -d
   docker compose logs -f musetalk          # wait for "Avatar stella ready"
   curl -s http://127.0.0.1:8000/health     # voice services + "avatar": {"status": "ready", ...}
   curl -s http://127.0.0.1:8000/api/avatar/status
   ```

5. Open the page. The orchestrator is published on `127.0.0.1:8000` of the GPU
   host; from another computer use an SSH port forward
   (`ssh -N -L 8000:127.0.0.1:8000 user@gpu-host`) and open
   http://127.0.0.1:8000. The badge reads **Avatar: ready** and replies play as
   Stella's video.

Stop with `docker compose down`. Weights (`musetalk/models/`) and prepared
avatars (volume `musetalk-results`) are kept.

The weights land in `musetalk/models/`, laid out exactly as in the official
README:

```text
models/
├── musetalk/        musetalk.json, pytorch_model.bin
├── musetalkV15/     musetalk.json, unet.pth        <- used by this service
├── syncnet/         latentsync_syncnet.pt
├── dwpose/          dw-ll_ucoco_384.pth
├── face-parse-bisent/ 79999_iter.pth, resnet18-5c106cde.pth
├── sd-vae/          config.json, diffusion_pytorch_model.bin
└── whisper/         config.json, pytorch_model.bin, preprocessor_config.json
```

If `gdown` fails on the Google Drive file (quota), download
`79999_iter.pth` manually from
https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812 into
`musetalk/models/face-parse-bisent/`. For a Hugging Face mirror, set
`HF_ENDPOINT=https://hf-mirror.com` in `.env` (the official script's default).

## Settings (root `.env`)

| Variable | Default | Purpose |
|---|---|---|
| `COMPOSE_PROFILES` | unset | `gpu` includes the musetalk service |
| `MUSETALK_ENABLED` | `false` | orchestrator calls MuseTalk |
| `MUSETALK_SERVICE_URL` | `http://musetalk:8010` | Compose service name; change only if MuseTalk runs elsewhere |
| `MUSETALK_AVATAR_ID` | `stella` | `avatars/<id>/`; also prepared at startup |
| `MUSETALK_API_KEY` | empty | optional shared key; both services read it from the same `.env` |
| `MUSETALK_BATCH_SIZE` | `20` | lower (e.g. 8) on out-of-memory |
| `MUSETALK_GPU_ID` | `0` | which GPU |
| `MUSETALK_MAX_AUDIO_SECONDS` | `60` | longest reply accepted |
| `MUSETALK_PREPARE_ON_START` | `true` | prepare the default avatar at startup |
| `MUSETALK_TIMEOUT` | `180` | orchestrator wait per video |
| `HF_ENDPOINT` | `https://huggingface.co` | weights download source |

## Network and security

- The service has **no published port**: `expose: 8010` makes it reachable only
  by other containers on the Compose network. The browser never talks to it;
  the orchestrator downloads each MP4, deletes it on the service and serves its
  own copy at `/api/avatar/videos/<id>.mp4`.
- The orchestrator stays bound to `127.0.0.1:8000` and has no authentication.
  Reach it through an SSH tunnel or a private VPN, not the public internet.
- `MUSETALK_API_KEY` is optional defence in depth inside the Compose network.

## Service API (internal, port 8010)

When `MUSETALK_API_KEY` is set, every endpoint requires
`Authorization: Bearer <key>`.

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | status, `cuda_available`, `gpu_available`, GPU name/VRAM, torch/CUDA versions, `model_loaded`; `200` only when ready |
| GET | `/avatars` | avatars under `avatars/` and whether they are prepared |
| POST | `/avatars/{id}/prepare` | prepare an avatar ahead of its first request |
| POST | `/generate` | multipart `audio` (PCM WAV, ≤ 20 MB, ≤ 60 s), optional `avatar_id` → `filename`, `video_url`, timings |
| GET | `/videos/{id}.mp4` | download (H.264 + AAC, faststart) |
| DELETE | `/videos/{id}.mp4` | the orchestrator deletes each video after copying it |

Errors: `400` bad audio or avatar, `401` key, `413` too large, `503` not ready,
`500` generation failure. Jobs run one at a time on the GPU. Inputs are deleted
after each request and outputs after `MUSETALK_OUTPUT_TTL_MINUTES` (30).

Calling it directly from the GPU host (it has no host port, so go through a container):

```bash
docker compose exec musetalk python -c "import urllib.request as u; print(u.urlopen('http://localhost:8010/health').read().decode())"
```

## CPU-only machines (the dev laptop)

Leave `COMPOSE_PROFILES` unset and `MUSETALK_ENABLED=false`. A plain
`docker compose build` / `up -d` then ignores the musetalk service; the voice
conversation works and the page reports the avatar as disabled.

MuseTalk is not usable on a CPU. On the laptop (i3-10110U, AVX2, no AVX-512/AMX)
the MuseTalk 1.5 UNet + VAE decoder measured about **10 s per frame**, about
250× slower than real time: roughly 21 minutes of rendering for a 5-second
reply. bf16 was about 200× slower again. The real-time code also hard-codes
fp16 and `cuda:0`. If the musetalk container is started without a GPU anyway,
`/health` reports `"CUDA is not available"` and the orchestrator falls back to
voice-only.

## Troubleshooting

- `could not select device driver "nvidia"` on `up`: the NVIDIA Container
  Toolkit is missing (step 1), or the musetalk profile is enabled on a CPU-only
  machine.
- `CUDA is not available` in `/api/avatar/status`: the container cannot see
  the GPU. Check `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`.
- `MuseTalk weights missing`: run `download_weights.sh` (step 3).
- `Avatar 'stella' has no reference file`: add `avatars/stella/reference.mp4`.
- Avatar `unavailable` with "MuseTalk is loading" / "preparing_avatar": first
  start; wait for "Avatar stella ready" in the log.
- Poor mouth region or face not found: use a frontal, well-lit reference
  video at 720p or above with an unobstructed face.
- Out of GPU memory: lower `MUSETALK_BATCH_SIZE` (e.g. 8).
- Long replies time out: raise `MUSETALK_TIMEOUT`.
