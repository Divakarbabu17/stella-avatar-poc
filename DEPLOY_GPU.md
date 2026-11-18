# GPU deployment on AI1 (`compose.gpu.yaml`)

Runs the complete Stella stack on one NVIDIA GPU server, with **every model on the GPU**.
The CPU laptop workflow (`compose.yaml`, `.env.example`) is unchanged and independent.

> **Status: not yet run on AI1.** Everything below is configured and written to the versions
> verified on paper (see "Compatibility"), but no container of this file has been built or
> started on the L40 yet. Treat the first run as the test; record the results in step 6.

## What runs where

| Container | Image / build | Python / PyTorch / CUDA | Port (internal) | GPU |
|---|---|---|---|---|
| `qwen` | `vllm/vllm-openai:v0.33.0` (pinned by digest) | 3.12 / vLLM's own / CUDA 13.0.3 | 8002 (OpenAI API) | yes, capped share |
| `whisper` | `Dockerfile` target `whisper`, GPU build args | 3.11 / 2.6.0 / cu126 wheels | 8001 | yes, FP16 |
| `kokoro` | `Dockerfile` target `kokoro`, GPU build args | 3.11 / 2.6.0 / cu126 wheels | 8003 | yes |
| `musetalk` | `musetalk/Dockerfile` | 3.10 / 2.0.1 / CUDA 11.8 | 8010 | yes |
| `orchestrator` | `Dockerfile` target `orchestrator` | 3.11 / none | **8000 → `127.0.0.1:8000`** | no |

Four different CUDA/PyTorch stacks, so four separate model containers: they cannot share an image.
Internal URLs are identical to the CPU stack (`http://whisper:8001`, `http://qwen:8002`,
`http://kokoro:8003`, `http://musetalk:8010`). The orchestrator talks to vLLM because
`QWEN_BACKEND=vllm` selects `VllmQwenClient` (`/v1/chat/completions`, history + system
prompt, `enable_thinking: false`).

## Compatibility (checked against vendor sources, not yet on hardware)

| Item | Requirement | AI1 |
|---|---|---|
| Driver for vLLM v0.33.0 (CUDA 13.0.3) | CUDA 13.x needs driver ≥ 580 | 596.36 = R595 branch (CUDA 13.2) ✔ |
| Driver for cu126 PyTorch (whisper, kokoro) | CUDA 12.6 | ✔ (backward compatible); torch 2.6.0 + cu126 already tested on AI1 |
| Driver for MuseTalk (CUDA 11.8) | driver ≥ 520 | ✔ (backward compatible) |
| L40 compute capability 8.9 in vLLM v0.33.0 build | arch list `7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0` | ✔ |
| L40 (sm_89) in PyTorch 2.0.1+cu118 (MuseTalk) | no native sm_89 kernels; runs sm_86 binaries (binary-compatible within 8.x) | expected ✔, **untested** |

## 1. Prerequisites on AI1

- Docker running **Linux containers** with GPU access (already verified on AI1).
- `docker compose version` → Compose v2.20 or newer.
- About **200 GB** free disk for images, weights and build cache (see the table below), and 64 GB RAM recommended.
- Run all commands from the repository root. They are written for a Linux shell (bash in WSL or a Linux VM).
  In PowerShell use `curl.exe` instead of `curl`, and backtick `` ` `` instead of `\` for line continuations.
- **Performance tip:** if Docker's Linux engine runs in WSL2, keep the repository (or at least
  `STELLA_MODELS_DIR` and `MUSETALK_MODELS_DIR`) on the Linux filesystem (e.g. `~/stella`),
  not on `C:\`. Bind mounts from NTFS make loading 20+ GB of weights much slower.

```bash
# GPU visible to containers with the same CUDA stacks the services use
docker run --rm --gpus all nvidia/cuda:13.0.3-base-ubuntu24.04 nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 2. Configuration

```bash
cp .env.gpu.example .env
# Set QWEN_API_KEY (required) and MUSETALK_API_KEY (recommended):
openssl rand -hex 32
```

Create or edit `.env` on AI1 itself, or save it with LF line endings. Windows line endings put a hidden `\r` into the keys.
`.env` is gitignored; never commit it.

## 3. Storage layout and model weights

Weights are **never** downloaded during `docker build`. They live in host folders mounted into the containers.

| Model | Size (approx.) | Host path | Downloaded by |
|---|---|---|---|
| Qwen3 `Qwen/Qwen3-8B-FP8` | ~9 GB | `models/huggingface/models--Qwen--Qwen3-8B-FP8/` | step 4b (or vLLM on first start) |
| Whisper `large-v3-turbo` | ~1.6 GB | `models/large-v3-turbo.pt` | step 4b (or on first start) |
| Kokoro-82M + voice `af_heart` | ~0.35 GB | `models/huggingface/models--hexgrad--Kokoro-82M/` | step 4b (or on first start) |
| MuseTalk 1.5 set (UNet, SD-VAE, whisper-tiny, DWPose, face-parsing, SyncNet, v1.0 UNet) | ~9–10 GB | `musetalk/models/` | step 4b, `download_weights.sh` (required) |
| vLLM compile cache | < 1 GB | Docker volume `stella-gpu_vllm-cache` | automatic |
| Prepared avatar (crops, latents, masks) | < 1 GB | Docker volume `stella-gpu_musetalk-results` | automatic, on first start |
| Generated avatar MP4s | small, TTL 60 min | `output/avatar_videos/` | runtime |

`models/` and `musetalk/models/` can be moved with `STELLA_MODELS_DIR` / `MUSETALK_MODELS_DIR` in `.env`.
Both are gitignored.

Docker images (approx.): MuseTalk ~15 GB, vLLM ~10–20 GB, whisper + kokoro GPU images ~7–9 GB together (shared PyTorch base), orchestrator < 1 GB.

## 4. Images: build and make them available on AI1

Images and tags used by `compose.gpu.yaml`:

| Image | Built from | Approx. size |
|---|---|---|
| `stella-orchestrator:gpu` | `Dockerfile` target `orchestrator` | < 1 GB |
| `stella-whisper:gpu` | `Dockerfile` target `whisper` + CUDA build args | ~7 GB (shares base with kokoro) |
| `stella-kokoro:gpu` | `Dockerfile` target `kokoro` + CUDA build args | ~8 GB (shares base with whisper) |
| `stella-musetalk:1.5` | `musetalk/Dockerfile` | ~15 GB |
| `vllm/vllm-openai:v0.33.0` | official, pulled (digest `sha256:d825baf9…a84c6e`) | ~10–20 GB |

Building needs no GPU, but **a successful build does not prove the image runs on the L40**.
That is only shown by steps 6–9 on AI1.

### Option A: build on the development machine (Windows, Docker Desktop) and transfer

PowerShell, from the repository root on the development machine:

```powershell
docker build --target orchestrator -t stella-orchestrator:gpu .
docker build --target whisper --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 --build-arg TORCH_SPEC=torch==2.6.0 -t stella-whisper:gpu .
docker build --target kokoro  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 --build-arg TORCH_SPEC=torch==2.6.0 -t stella-kokoro:gpu .
docker build -t stella-musetalk:1.5 ./musetalk        # 20-40 min the first time

# Build-time sanity checks (no GPU needed; they only prove the packages import)
docker run --rm stella-whisper:gpu python -c "import torch, whisper; print(torch.__version__, torch.version.cuda)"    # expect 2.6.0+cu126 12.6
docker run --rm stella-kokoro:gpu  python -c "import torch, kokoro, transformers; print(torch.__version__, transformers.__version__)"
docker run --rm stella-musetalk:1.5 python -c "import torch, mmcv, mmpose; print(torch.__version__, torch.version.cuda, mmcv.__version__)"   # expect 2.0.1+cu118 11.8 2.0.1
docker run --rm stella-orchestrator:gpu python -c "import app.orchestrator.main, app.orchestrator.serve; print('ok')"
docker image ls "stella-*"

# One archive; layers shared between images are stored once
docker save -o stella-images.tar stella-orchestrator:gpu stella-whisper:gpu stella-kokoro:gpu stella-musetalk:1.5
```

Copy `stella-images.tar` and the repository folder to AI1, using whatever file transfer you normally use. For example:

```powershell
robocopy . \\AI1\stella-transfer stella-images.tar /J            # SMB share, unbuffered copy for a large file
scp stella-images.tar <user>@AI1:/home/<user>/stella/             # or SSH
```

The repository must be on AI1 too: compose file, `.env`, `avatars/`, and the `models/` / `output/` folders. Copy it without `models/`, `musetalk/models/` and `output/` content.

On AI1, from the repository root:

```bash
docker load -i stella-images.tar
docker image ls "stella-*"
# vLLM: pull directly on AI1 (keeps the pinned registry digest)
docker pull vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e
```

If AI1 has no access to Docker Hub, move vLLM the same way. On the development machine:

```powershell
docker pull vllm/vllm-openai:v0.33.0
docker image inspect vllm/vllm-openai:v0.33.0 --format "{{json .RepoDigests}}"   # must contain sha256:d825baf9ce4b...a84c6e
docker save -o vllm-v0.33.0.tar vllm/vllm-openai:v0.33.0
```

Then on AI1, run `docker load -i vllm-v0.33.0.tar` and set `VLLM_IMAGE=vllm/vllm-openai:v0.33.0` in `.env`. A loaded image has no registry digest, so the pinned reference would otherwise try to pull.

Always start with `--no-build` on AI1 so Compose uses the loaded images and never rebuilds:
`docker compose -f compose.gpu.yaml up -d --no-build ...` (step 6).

**After a code or UI change** only the affected image needs rebuilding and moving. UI changes only touch `stella-orchestrator:gpu`, which is small:

```powershell
docker build --target orchestrator -t stella-orchestrator:gpu .
docker save -o stella-orchestrator.tar stella-orchestrator:gpu
```

Then on AI1: `docker load -i stella-orchestrator.tar`, followed by `docker compose -f compose.gpu.yaml up -d --no-build orchestrator`. Only the orchestrator restarts; models stay loaded.

### Option B: build directly on AI1

```bash
docker compose -f compose.gpu.yaml pull qwen
docker compose -f compose.gpu.yaml build          # MuseTalk image: 20-40 min the first time
```

## 4b. Pre-download the model weights on AI1 (no GPU memory used yet)

```bash

# --no-deps: do not start vLLM for these one-off download containers
docker compose -f compose.gpu.yaml run --rm --no-deps --entrypoint python3 qwen -c \
  "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-8B-FP8'))"
docker compose -f compose.gpu.yaml run --rm --no-deps whisper python -c \
  "import whisper; whisper.load_model('large-v3-turbo', device='cpu', download_root='/app/models'); print('ok')"
docker compose -f compose.gpu.yaml run --rm --no-deps kokoro python -c \
  "from huggingface_hub import hf_hub_download as d; [print(d('hexgrad/Kokoro-82M', f, cache_dir='/app/models/huggingface')) for f in ('config.json', 'kokoro-v1_0.pth', 'voices/af_heart.pt')]"
docker compose -f compose.gpu.yaml run --rm --no-deps musetalk bash /opt/stella/download_weights.sh
```

If you change `QWEN_MODEL` or `WHISPER_MODEL`, download that name instead.
After this step you may set `HF_HUB_OFFLINE=1` in `.env`.
If `gdown` fails on the Google Drive face-parsing file (quota), see [musetalk/README.md](musetalk/README.md).

## 5. Stella reference video (required for the avatar)

Put it on AI1 at **`avatars/stella/reference.mp4`**. It is gitignored (personal likeness) and is not in the repository.
Optionally add `avatars/stella/idle.png`, which the page shows while Stella is idle.

- 5–15 s, one person, face clearly visible and roughly frontal, neutral or closed mouth, little head movement.
- Even lighting, 720p or 1080p, nothing covering the mouth or jaw.
- **Use only footage of a person who has consented** to being used as Stella.
- `reference.png` works too, but then only the mouth moves.

Details: [avatars/stella/README.md](avatars/stella/README.md). After replacing the video, restart the musetalk container. It re-prepares the avatar automatically.

## 6. Start in order and measure VRAM

Do this one service at a time the first time: the memory budget below is an **estimate**
until these numbers are recorded on AI1.

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv          # A: baseline

docker compose -f compose.gpu.yaml up -d --no-build qwen
docker compose -f compose.gpu.yaml ps qwen                                 # wait for (healthy)
docker compose -f compose.gpu.yaml logs qwen | grep -iE "memory|KV cache|kernel|error"
nvidia-smi --query-gpu=memory.used --format=csv                            # B: + vLLM

docker compose -f compose.gpu.yaml up -d --no-build whisper   && sleep 60 && nvidia-smi --query-gpu=memory.used --format=csv   # C
docker compose -f compose.gpu.yaml up -d --no-build kokoro    && sleep 60 && nvidia-smi --query-gpu=memory.used --format=csv   # D
docker compose -f compose.gpu.yaml up -d --no-build musetalk
docker compose -f compose.gpu.yaml logs -f musetalk                        # wait for "Avatar stella ready", then Ctrl+C
nvidia-smi --query-gpu=memory.used --format=csv                            # E
docker compose -f compose.gpu.yaml up -d --no-build orchestrator
# (Option B, built on AI1: the same commands work; --no-build just skips rebuilding.)

# F: peak during real work. Leave this running in a second terminal, then run the step 7 tests
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 1
```

| Point | Expected (estimate) | Measured on AI1 |
|---|---|---|
| B − A (vLLM, util 0.35) | ~16 GB (fixed budget) | |
| C − B (Whisper large-v3-turbo FP16) | ~2–3 GB | |
| D − C (Kokoro) | ~0.5–1 GB | |
| E − D (MuseTalk, batch 20, avatar prepared) | ~5–10 GB | |
| F peak during a turn with avatar | total ≲ 41 GB (90 %) | |

If peak usage goes above ~90 % of 46 GB: lower `MUSETALK_BATCH_SIZE` (e.g. 8), use a smaller
`WHISPER_MODEL`, or lower `VLLM_MAX_MODEL_LEN` / `VLLM_GPU_MEMORY_UTILIZATION`. Raise vLLM's share only if there is clear headroom.

Notes:
- vLLM claims its whole share at startup and refuses to start if that much is not free.
  That is why the other model services wait for `qwen` to be healthy. If vLLM is restarted
  later, the memory the others hold must still leave its share free.
- Under WSL2, `nvidia-smi` does not show per-process memory. Use the differences above,
  vLLM's log lines, and each PyTorch service's own `/health` fields
  (`gpu_memory_reserved_mb`, `gpu_memory_peak_reserved_mb`; add ~0.3–0.5 GB per process for the CUDA context).

## 7. Verify GPU access and inference, per service

```bash
# --- GPU visible inside each container ---
for s in whisper kokoro musetalk; do
  docker compose -f compose.gpu.yaml exec $s python -c \
    "import torch; print('$s', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
done
docker compose -f compose.gpu.yaml exec qwen nvidia-smi

# --- Readiness of everything, as the orchestrator sees it ---
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
#   services.whisper / services.kokoro: "status": "ready", "device": "cuda", "gpu": "NVIDIA L40", gpu_memory_*_mb
#   services.qwen: "backend": "vllm", "status": "ready", "served_models": ["Qwen/Qwen3-8B-FP8"]
#   avatar: "status": "ready", "cuda_available": true, "gpu": {"name": "NVIDIA L40", ...}

# --- vLLM directly (API key; /health has an empty body by design) ---
docker compose -f compose.gpu.yaml exec qwen python3 -c "import os,json,urllib.request as u; \
print(json.load(u.urlopen(u.Request('http://localhost:8002/v1/models', headers={'Authorization': 'Bearer ' + os.environ['VLLM_API_KEY']}))))"

# --- Qwen3 inference (through the orchestrator) ---
curl -s -X POST http://127.0.0.1:8000/api/qwen/generate -H "Content-Type: application/json" \
  -d '{"prompt": "Hi Stella, introduce yourself in one sentence.", "max_new_tokens": 48}'
# expect a reply with no <think> text; check generation_seconds and generated_tokens

# --- Kokoro inference: creates a WAV under output/api/ ---
curl -s -X POST http://127.0.0.1:8000/api/kokoro/synthesize -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is Stella speaking from the GPU server."}'
# note "audio_file", e.g. output/api/kokoro_20261118_101500_ab12cd34.wav
WAV=$(ls -t output/api/kokoro_*.wav | head -1)

# --- Whisper inference on that WAV (round trip: expect the same sentence back) ---
curl -s -F "file=@$WAV" http://127.0.0.1:8000/api/whisper/transcribe

# --- MuseTalk only, with that WAV ---
curl -s -F "file=@$WAV" -F "avatar_id=stella" http://127.0.0.1:8000/api/avatar/generate
# expect "status": "ready", a video_url and musetalk_render_seconds

# --- Full pipeline: Whisper -> Qwen3 -> Kokoro -> MuseTalk ---
curl -s -F "file=@$WAV" -F "history=[]" http://127.0.0.1:8000/api/conversation/turn \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['transcript'], '|', d['response'], '|', d['avatar']['status'], d['timings'])"
```

**Browser:** the UI listens on `127.0.0.1:8000` only, because it has no authentication. Open
http://localhost:8000 on AI1 itself (e.g. over RDP), or forward the port:
`ssh -N -L 8000:127.0.0.1:8000 <user>@AI1`. Browsers allow the microphone only on `localhost`
or `https://`, so `http://<AI1-ip>:8000` would not get mic access even if published.

## 8. The orchestrator UI and manual tests

Open http://localhost:8000 (or `https://<AI1>:8000` with HTTPS, step 9). The page has three parts:

- **System status** (top, both views; refreshes every 10 s):
  - One card per service: whether its container answers on the Docker network, model readiness (loading / ready / error with the message), device, GPU memory held by that model's PyTorch process (vLLM: its fixed budget, KV-cache use and request queue), load time, uptime, last processing time.
  - A **GPU** card: whole-GPU utilization, VRAM used / total, temperature and power, read with `nvidia-smi` inside the whisper or kokoro container.
  - **Recent processing times**: last / avg / min / max of the last 20 runs per stage, including the Stella Chat turn total.
  - Data source: `GET /api/system/status`.
- **AI Lab**: independent tests. Each one calls only its own model container; the others can be stopped.
- **Stella Chat**: the complete Whisper → Qwen3 → Kokoro → MuseTalk conversation (unchanged behaviour).

"Container: not reachable" means stopped, crashed or still starting (vLLM opens its port only after loading).
The orchestrator deliberately has no access to the Docker socket, so `docker compose -f compose.gpu.yaml ps` is the authoritative view.

### Whisper (AI Lab → Whisper)
1. Press **● Start recording** and allow the microphone. Say a known sentence, e.g. "The quick brown fox jumps over the lazy dog." Press **■ Stop and transcribe**.
2. Expect a card showing the transcript, **Whisper processing** time, round trip, and a player for your recording.
3. Also test **upload a file** (e.g. a WAV from the Kokoro test) and an empty or silent recording. The latter should give a clear error, not a crash.

### Qwen3 (AI Lab → Qwen3)
1. Enter a prompt, e.g. "In two sentences, what can you help me with?", and press **Generate**.
2. Expect:
   - the response, with no `<think>` text;
   - **Time to first token**;
   - **Tokens / s (overall)** and **(decode)**;
   - tokens, finish reason, and backend `vllm`.
3. Try a custom system prompt and a small max-new-tokens value (finish reason should become `length`).

### Kokoro (AI Lab → Kokoro)
1. Enter any text, pick a voice and speed, and press **Synthesize**. Qwen3 is not involved.
2. Expect the audio to play in the browser, with **Kokoro synthesis** time, audio length, real-time factor, and a WAV download link.

### Independence check
While one model is being tested, stop the others:

```bash
docker compose -f compose.gpu.yaml stop musetalk kokoro         # then use AI Lab → Whisper / Qwen3: still work
docker compose -f compose.gpu.yaml start kokoro musetalk        # no restart of anything else needed
```

The stopped services show **not reachable** in System status, and their AI Lab panels return a clear "service is not ready" error.

### Complete pipeline (Stella Chat)
1. Wait until System status shows 4/4 ready and the avatar badge says **Avatar: ready**.
2. Press the microphone (or Space), ask a question, and press it again.
3. Expect:
   - your transcript, then Stella's reply, in the conversation;
   - the lip-synced avatar video playing, with its audio;
   - **Last turn**: Whisper, Qwen3, Kokoro, MuseTalk render, avatar total, server total, round trip, and **Stop → playback** (end-to-end latency as you experience it).
4. Ask a follow-up that depends on the first answer, to confirm the conversation history is used.
5. **Fallback test:** `docker compose -f compose.gpu.yaml stop musetalk`, then ask again. Stella should answer with voice only, plus an "Avatar video unavailable" notice. Bring it back with `start musetalk`; it re-prepares the avatar (see its log).
6. Record per-turn timings and the GPU card's peak VRAM in the step 6 table.

## 9. HTTPS (optional: microphone from other machines)

Browsers allow the microphone only on `https://` or `http://localhost`. To let testers open the UI from their own PCs without an SSH tunnel:

```bash
mkdir -p certs
# Self-signed certificate; put AI1's real host name and IP in subjectAltName
docker run --rm -v "$PWD/certs:/certs" alpine/openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout /certs/stella.key -out /certs/stella.crt -subj "/CN=AI1" \
  -addext "subjectAltName=DNS:AI1,DNS:localhost,IP:<AI1-IP>"
```

In `.env`:

```ini
STELLA_TLS_CERTFILE=/app/certs/stella.crt
STELLA_TLS_KEYFILE=/app/certs/stella.key
STELLA_BIND_ADDRESS=0.0.0.0
```

Then run `docker compose -f compose.gpu.yaml up -d --no-build orchestrator` and open `https://<AI1>:8000`.

- Browsers warn about a self-signed certificate. To remove the warning on Windows clients, import `stella.crt` into the trusted root store: `Import-Certificate -FilePath stella.crt -CertStoreLocation Cert:\CurrentUser\Root`.
- **The page still has no login.** Only expose it on a trusted internal network, and restrict port 8000 in the Windows firewall to the testers' addresses.
- If Docker's Linux engine runs inside WSL2 with NAT networking, a published port may be reachable only from AI1 itself. Use WSL "mirrored" networking or a `netsh interface portproxy` rule for LAN access.

## 10. Operations

```bash
docker compose -f compose.gpu.yaml ps                       # health of every service
docker compose -f compose.gpu.yaml logs -f <service>
docker compose -f compose.gpu.yaml restart musetalk         # e.g. after replacing reference.mp4
docker compose -f compose.gpu.yaml down                     # stop; weights and volumes are kept
```

All services use `restart: unless-stopped`.

## Unresolved compatibility risks

1. **Nothing has run on AI1 yet.** The code changes were written on a machine with no Python, Docker or GPU, so they have not been executed anywhere.
2. **MuseTalk** (`musetalk/Dockerfile`) has never been built. Things to watch:
   - `mim install mmcv==2.0.1` needs a prebuilt wheel for torch 2.0 / cu118, or it compiles from source (slow but possible on the `-devel` base).
   - PyTorch 2.0.1+cu118 has no sm_89 kernels and relies on running sm_86 binaries on the L40.
   - `download_weights.sh` depends on the `huggingface-cli` and `gdown` versions pinned by MuseTalk's own requirements, and on Google Drive quotas.
   - MuseTalk's VRAM at batch 20 has not been measured.

   Test: steps 4–7, the `musetalk` row in each.
3. **Qwen3 FP8 on Ada.** On the L40, vLLM serves block-quantized FP8 checkpoints through a fallback kernel (Marlin or Triton), not a native FP8 kernel. Memory savings remain; speed may be lower than expected. The selected kernel appears in the `qwen` log. Fallback option: `QWEN_MODEL=Qwen/Qwen3-8B` (BF16) with `VLLM_GPU_MEMORY_UTILIZATION=0.45`.
4. **vLLM v0.33.0 is recent** (2026-11-02). It is pinned by digest so it cannot change underneath you; upgrade deliberately.
5. **WSL2 host specifics** (if AI1's Linux engine runs in WSL2):
   - vLLM disables pinned memory under WSL by default (slightly slower host↔GPU copies).
   - Per-process VRAM is not visible in `nvidia-smi`.
   - NTFS bind mounts are slow.
6. **GPU memory is shared, not partitioned.** The PyTorch services' caching allocators keep their peak memory reserved, and MuseTalk spikes while preparing an avatar. Only the step 6 measurements show the real headroom.
7. **The whisper and kokoro GPU images** (torch 2.6.0+cu126 with `openai-whisper`, `kokoro==0.9.4`, `transformers==4.57.1`, numpy 2.4.6) have not been built. The Dockerfile pins the installed torch through `PIP_CONSTRAINT`, so any package that demands a different torch fails the build loudly instead of silently swapping it.
8. **Startup dependency.** If `qwen` (vLLM) fails to become healthy, `whisper`, `kokoro` and `musetalk` are not started (intended: vLLM must claim its memory first). The orchestrator still starts and reports what is missing.
9. **GPU metrics in the UI** need `nvidia-smi` inside the whisper or kokoro container. `NVIDIA_DRIVER_CAPABILITIES=compute,utility` is set for that; under WSL2 it is looked up in `/usr/lib/wsl/lib`. If it is still missing, the GPU card says so; use `nvidia-smi` on the host instead. Per-container VRAM is PyTorch-reserved memory (plus ~0.3–0.5 GB of CUDA context each), and for vLLM its configured budget, not a per-process measurement.
10. **Qwen3 timing figures** (time to first token, tokens/s) are measured by the orchestrator over the internal Docker network while it reads vLLM's stream. They include a few milliseconds of network overhead.
11. **Unchanged limits:** one conversation turn at a time; the avatar video is rendered after the full reply (no streaming); no authentication on the UI.
