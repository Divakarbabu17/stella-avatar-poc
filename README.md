# Stella Voice + Avatar POC

Standalone local proof of concept for a self-hosted Stella assistant (the
eventual HeyGen replacement). You talk to Stella in the browser; she answers
with her voice and, on a host with an NVIDIA GPU, a lip-synced video avatar.

- **Speech → text:** Whisper
- **Reply:** Qwen3
- **Text → speech:** Kokoro
- **Talking face:** MuseTalk 1.5

All five services are in one `compose.yaml`. The MuseTalk service is under the
Compose profile `gpu`:

- **CPU-only laptop (development):** four services, voice-only. MuseTalk is
  neither built nor started.
- **NVIDIA GPU host (deployment):** all five services with `COMPOSE_PROFILES=gpu`
  and `MUSETALK_ENABLED=true`. Whisper, Qwen3 and Kokoro still run on CPU;
  MuseTalk uses the GPU.

Not in scope yet: LiveKit/WebRTC, real-time streaming, interruption handling,
integration with the Stella production application.

## Architecture

```text
 One Docker host (compose.yaml)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Browser  http://127.0.0.1:8000                                           │
 │   mic (MediaRecorder) → upload; plays avatar MP4 (or WAV)                │
 │            │                       ▲                                     │
 │ ┌──────────▼───────────────────────┴───────────────┐                     │
 │ │ orchestrator :8000 (FastAPI, no ML)              │  only published     │
 │ │ web page, /api/conversation/turn,                │  port (127.0.0.1)   │
 │ │ /api/avatar/videos/<id>.mp4 (local copies)       │                     │
 │ └───┬──────────────┬──────────────┬──────────────┬─┘                     │
 │     │       internal Compose network (not published)                     │
 │ ┌───▼────┐   ┌─────▼────┐   ┌─────▼────┐   ┌─────▼──────────────────┐    │
 │ │whisper │   │ qwen     │   │ kokoro   │   │ musetalk :8010         │    │
 │ │ :8001  │   │ :8002    │   │ :8003    │   │ profile "gpu", NVIDIA  │    │
 │ │ CPU    │   │ CPU      │   │ CPU      │   │ MuseTalk 1.5 fp16      │    │
 │ └────────┘   └──────────┘   └──────────┘   └────────────────────────┘    │
 │   ./models (cache)                   musetalk/models, avatars/, results  │
 └──────────────────────────────────────────────────────────────────────────┘
```

One conversation turn:

1. The browser records your question and uploads it to `POST /api/conversation/turn`.
2. Whisper transcribes it. FFmpeg in the whisper container decodes any browser format.
3. Qwen3 replies, using the last 6 messages of the session.
4. Kokoro speaks the reply as a 24 kHz WAV.
5. If MuseTalk is enabled, the orchestrator sends that WAV to the musetalk
   service, downloads the finished MP4 (Kokoro audio muxed in), deletes it there
   and serves its own copy at `/api/avatar/videos/<id>.mp4`.
6. The page plays the avatar video. If there is no video (disabled, service
   not running or generation failed), it plays the Kokoro audio instead. A
   MuseTalk problem never fails the turn.

**A/V sync.** The MP4 contains the Kokoro audio that drove the lip movement, so
the page plays only that one file, never a video and a separate audio track
together. Voice and lips cannot drift apart.

Design points:

- **One model per container, loaded once.** `/health` shows `load_count: 1`.
- **Private model services.** Only the orchestrator is published, and only on
  `127.0.0.1`.
- **Start what you need.** The orchestrator checks each service's `/health`
  per request and returns a clear 503 otherwise.
- **Movable to another machine.** Each model service is plain HTTP behind a URL
  setting (`WHISPER_URL`, `QWEN_URL`, `KOKORO_URL`, `MUSETALK_SERVICE_URL`).

## Project layout

```text
app/
  config.py, schemas.py
  services/       model logic (whisper, qwen, kokoro) + voice.py pipeline
  servers/        one FastAPI app per model container
  orchestrator/
    main.py       public API + web page
    clients.py    HTTP clients for whisper/qwen/kokoro
    avatar.py     MuseTalk client, video store, avatar status
    static/       index.html, app.js (conversation), lab.js (AI Lab), style.css (no framework, no build step)
  main.py qwen.py kokoro.py pipeline.py   debugging CLIs
musetalk/         musetalk service: Dockerfile (CUDA), service/, download_weights.sh, README.md
avatars/stella/   avatar reference (MuseTalk) + optional idle image (page)
tests/            musetalk_contract_stub.py (test double, see Testing)
requirements/     per-service pinned dependencies
Dockerfile        orchestrator, whisper, qwen, kokoro, cli images
compose.yaml      all services (musetalk under profile "gpu")
```

## Models and configuration

| Service | Model | Where | Default config |
|---|---|---|---|
| whisper | Whisper `base` | CPU | `fp16=False` |
| qwen | `Qwen/Qwen3-0.6B` | CPU | float32, thinking off, temp 0.7 / top-p 0.8 / top-k 20 |
| kokoro | `hexgrad/Kokoro-82M` | CPU | voice `af_heart`, 24 kHz mono WAV |
| musetalk | MuseTalk 1.5 (`musetalkV15/unet.pth`) | NVIDIA GPU | fp16, 25 fps, batch 20 |

Settings live in `.env` (copy `.env.example`). The main ones:

| Variable | Default | Purpose |
|---|---|---|
| `COMPOSE_PROFILES` | unset | `gpu` adds the musetalk service (GPU host only) |
| `MUSETALK_ENABLED` | `false` | `true` turns on the avatar video |
| `MUSETALK_SERVICE_URL` | `http://musetalk:8010` | the Compose service; change only if MuseTalk runs elsewhere |
| `MUSETALK_API_KEY` | empty | optional; shared by orchestrator and musetalk via the same `.env` |
| `MUSETALK_TIMEOUT` | `180` | seconds to wait for one video |
| `MUSETALK_AVATAR_ID` | `stella` | folder under `avatars/` |
| `AVATAR_VIDEO_TTL_MINUTES` | `60` | how long generated videos stay in `output/avatar_videos/` |
| `CONVERSATION_HISTORY_MESSAGES` | `6` | earlier messages sent to Qwen3 |
| `CONVERSATION_MAX_NEW_TOKENS` | `96` | reply length cap for voice turns |

MuseTalk-specific settings (batch size, GPU id, limits) are listed in
[musetalk/README.md](musetalk/README.md).

### Stella's avatar

`avatars/stella/` holds the avatar. MuseTalk needs `reference.mp4`
(recommended; looped while she speaks, re-encoded to 25 fps) or
`reference.png` (supported, but only the mouth moves). The page optionally
uses `idle.png` as the still image shown while idle or processing; otherwise it
shows a neutral placeholder. To replace the avatar, swap `reference.mp4` and run
`docker compose restart musetalk`. It notices the change and re-prepares
automatically. Details: [avatars/stella/README.md](avatars/stella/README.md).
**No reference file is in the repository yet.**

## Run it on the laptop (voice-only)

From this directory in PowerShell:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose build
docker compose up -d
curl.exe -s http://127.0.0.1:8000/health      # wait for "status":"ok" (about a minute after start)
```

Open **http://127.0.0.1:8000** in Chrome, Edge or Firefox:

1. Press the microphone button (or Space) and allow microphone access.
2. Ask a question and press the button again to send.
3. Stella's reply plays automatically. Transcript, reply and timings appear on the page.

Controls on the page:

- **Stop playback** and **Replay**.
- **Voice** selector and **Clear conversation**.
- **Avatar video** toggle: switch it off for voice-only replies even when MuseTalk is enabled.
- An avatar status badge.

History is kept per browser tab (`sessionStorage`).

### AI Lab (test each model on its own)

The **AI Lab** tab (http://127.0.0.1:8000/#lab) tests one model at a time.
Each request goes to that model's service only, and the conversation and its
history are not touched. Every panel shows the service status, the inputs, a
run button, and one result card per run with its timings or the error.

| Panel | Input | Output | Endpoint |
|---|---|---|---|
| Whisper | any audio file | transcript, Whisper time | `POST /api/whisper/transcribe` |
| Qwen3 | prompt, optional system prompt, optional max new tokens | response, tokens, tokens/s, generation time | `POST /api/qwen/generate` |
| Kokoro | text, voice, speed | playable WAV, length, format, synthesis time | `POST /api/kokoro/synthesize` |
| MuseTalk | avatar folder + PCM WAV (or "Use latest Kokoro audio") | MP4 with timings | `POST /api/avatar/generate` |

With MuseTalk disabled (the laptop default) or not running, the MuseTalk panel
shows that status instead of a video. Direct links: `#lab/whisper`,
`#lab/qwen`, `#lab/kokoro`, `#lab/musetalk`.

Stop everything with `docker compose stop` (or `docker compose down`). Models
stay cached in `models/`.

## Run the full stack with the avatar (NVIDIA GPU host)

Copy the whole project to the GPU host. Then, from the project root:

```bash
cp .env.example .env
# in .env:  COMPOSE_PROFILES=gpu   and   MUSETALK_ENABLED=true
docker compose build
docker compose run --rm musetalk bash /opt/stella/download_weights.sh
cp /path/to/stella.mp4 avatars/stella/reference.mp4
docker compose up -d
docker compose logs -f musetalk                 # wait for "Avatar stella ready"
curl -s http://127.0.0.1:8000/api/avatar/status  # service.status "ready"
```

Open http://127.0.0.1:8000 on the GPU host, or from your laptop through
`ssh -N -L 8000:127.0.0.1:8000 user@gpu-host`. The badge reads
**Avatar: ready**, and replies play as Stella's video.

Host preparation (NVIDIA driver, Container Toolkit), weights layout and
troubleshooting: [musetalk/README.md](musetalk/README.md). For voice-only on
the same host, set `MUSETALK_ENABLED=false` and run `docker compose up -d`.

## API (orchestrator, port 8000)

Interactive docs: http://127.0.0.1:8000/docs

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | the web page |
| GET | `/health` | voice services + `avatar` (MuseTalk) status |
| POST | `/api/conversation/turn` | multipart `file` (any browser audio), `history` (JSON), `voice`, `avatar` (`true`/`false`) → `transcript`, `response`, `audio_base64` (WAV), `avatar` {`status`, `video_url`, timings}, timings |
| POST | `/api/voice/process` | file-based pipeline (WAV saved to `output/api/`), plus `avatar` |
| POST | `/api/whisper/transcribe`, `/api/qwen/generate`, `/api/kokoro/synthesize` | individual stages |
| GET | `/api/avatar/status` | avatar enabled?, GPU service health, idle image |
| GET | `/api/avatar/avatars` | avatar folders under `avatars/` and their reference file (AI Lab) |
| POST | `/api/avatar/generate` | AI Lab: multipart `file` (PCM WAV) + optional `avatar_id` → MuseTalk only; `200` ready, `503` disabled/unavailable, `502` failed, `422` bad WAV/id |
| GET | `/api/avatar/videos/{id}.mp4` | generated avatar video (`video/mp4`, HTTP range support) |
| GET | `/api/audio/{name}.wav` | WAVs from the file-based endpoints |

`avatar.status` is one of:

| Status | Meaning |
|---|---|
| `ready` | `video_url` set |
| `disabled` | `MUSETALK_ENABLED=false` |
| `off` | the request asked for no video |
| `unavailable` | musetalk not running, still loading, wrong key, or no CUDA |
| `failed` | generation or download error |

## Testing

### Laptop commands (PowerShell)

Voice stack and individual endpoints:

```powershell
curl.exe -s http://127.0.0.1:8000/health
curl.exe -s -F "file=@output/jfk.wav" http://127.0.0.1:8000/api/whisper/transcribe
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/qwen/generate -ContentType "application/json" `
  -Body (@{ prompt = "Hi Stella, introduce yourself in one sentence."; max_new_tokens = 48 } | ConvertTo-Json)
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/kokoro/synthesize -ContentType "application/json" `
  -Body (@{ text = "Hello! Welcome to our AI assistant." } | ConvertTo-Json)
```

One conversation turn from a file, as the page sends it (shows the avatar status):

```powershell
$r = curl.exe -s -F "file=@output/jfk.wav" -F "history=[]" http://127.0.0.1:8000/api/conversation/turn | ConvertFrom-Json
$r | Format-List transcript, response, avatar, timings
```

If `output/jfk.wav` is missing:
`docker compose --profile cli run --rm cli ffmpeg -y -i samples/jfk.flac -ar 16000 -ac 1 output/jfk.wav`.

Avatar plumbing (orchestrator side):

```powershell
curl.exe -s http://127.0.0.1:8000/api/avatar/status
# After a turn with avatar.status = "ready":
curl.exe -s -o NUL -w "%{http_code} %{content_type}`n" "http://127.0.0.1:8000$($r.avatar.video_url)"
```

### What has been tested on the CPU laptop

| Check | Result |
|---|---|
| Page served at `/`, static assets, `/docs` | 200 |
| Browser-format recordings (webm/opus, ogg/opus) through `/api/conversation/turn` | transcribed correctly |
| 3-turn conversation with history | turn 3 correctly recalled turn 1's question |
| `MUSETALK_ENABLED=false` | `avatar.status: disabled`, voice unchanged |
| Enabled, GPU service unreachable | `unavailable`, voice still returned (+2 ms) |
| Real MuseTalk service code on CPU (no GPU) | `/health` 503 `cuda_available:false`, clear "CUDA is not available"; `/generate` 503; wrong/missing key 401 |
| Orchestrator → that service, wrong key / right key | `unauthorized` / `unavailable` with the CUDA message; voice still returned |
| Success path against `tests/musetalk_contract_stub.py` (test double, grey frame, **not MuseTalk**) | 3 turns `ready`; served MP4 200 `video/mp4`, range 206, H.264 + AAC, duration = Kokoro audio, faststart; video deleted on the "GPU" side every time |
| `/api/voice/process` with avatar on/off, individual endpoints | all 200 |
| AI Lab: Whisper / Qwen3 / Kokoro / MuseTalk(stub) requests, counted in each service's access log | each request reached only its own model; error cases return clear 422/503 |
| AI Lab: four requests (one per model) at the same time | all 200, one call per model |
| Conversation after adding the AI Lab | 3 turns with history; 3 calls per model; unchanged |
| MuseTalk 1.5 UNet + VAE decoder on this CPU (real architecture, random weights) | ~10 s per frame, ~250× slower than real time: not usable on CPU |

### Still needs the NVIDIA GPU host

- `docker compose build` of the musetalk image (not built on the laptop) and the weights download.
- `/health` reporting `ready` with CUDA.
- Avatar preparation from your reference video.
- Real lip-synced MP4s, their quality, and MuseTalk generation time.
- Audio/video sync as perceived in the browser.
- Stella visibly speaking on the page.
- Repeated turns with real videos.

The browser microphone flow was exercised by you. My tests above used recorded
files in the same formats browsers send.

## Measured performance (CPU laptop)

| Stage | Typical (warm) |
|---|---|
| Whisper (short question) | 3-6 s |
| Qwen3 (≤ 96 tokens) | 3-6 s |
| Kokoro (2-4 s of speech) | 2-5 s |
| Voice turn total | 8-16 s |
| Avatar step when unreachable/disabled | < 0.01 s |
| MuseTalk generation | not measured on a GPU yet; CPU ~10 s per frame |

Memory: about 4.5 GB for the four laptop containers.

## Limitations

- **Latency.** The video is generated after the full Kokoro WAV, and the page
  waits for it. MuseTalk time is added on top of the 8-16 s voice turn. There
  is no streaming yet.
- **One turn at a time.** Overlapping requests get `429`, and the GPU service
  also processes one job at a time.
- **Reference matters.** MuseTalk renders the face region at 256 × 256, and
  its own documentation notes some jitter and loss of fine lip detail. An image
  reference produces a mostly static face.
- **GPU support.** RTX 50-series is not supported by MuseTalk's pinned
  PyTorch 2.0.1 + CUDA 11.8 stack.
- **No authentication on the web app.** It is bound to `127.0.0.1` only; on a
  GPU host reach it through an SSH tunnel or a private VPN. The musetalk service
  has no published port at all.
- **Qwen3-0.6B reply quality.** Small model; replies are sometimes clumsy.

## Debugging CLIs

```powershell
docker compose --profile cli build cli
docker compose --profile cli run --rm cli python -m app.main --input output/jfk.wav
docker compose --profile cli run --rm cli python -m app.qwen --prompt "Hi Stella"
docker compose --profile cli run --rm cli python -m app.kokoro --text "Hello!" --voice af_heart
docker compose --profile cli run --rm cli python -m app.pipeline --input output/jfk.wav --max-new-tokens 48
```

## Host assessment (laptop)

Windows 11, Docker Desktop (WSL 2), 4 logical cores, 20 GiB RAM (Docker VM
9.6 GiB), Intel UHD Graphics, no NVIDIA GPU: CPU inference only.

## References

- Whisper: https://github.com/openai/whisper
- Qwen3 0.6B: https://huggingface.co/Qwen/Qwen3-0.6B
- Kokoro: https://github.com/hexgrad/kokoro
- MuseTalk: https://github.com/TMElyralab/MuseTalk
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
