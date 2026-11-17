"""MuseTalk engine: loads the official MuseTalk 1.5 models once and renders lip-synced MP4s.

This module drives the official `scripts/realtime_inference.py` from the MuseTalk
repository (https://github.com/TMElyralab/MuseTalk) instead of re-implementing it:

* Model loading mirrors that script's `__main__` block for `--version v15`
  (fp16 UNet/VAE/positional encoding, whisper-tiny audio encoder, FaceParsing).
* The script's `Avatar` class reads module-level globals (`args`, `vae`, `unet`,
  `pe`, `timesteps`, `audio_processor`, `whisper`, `weight_dtype`, `device`,
  `fp`); they are injected into the imported module after loading.
* `Avatar(preparation=True)` prompts with input() when the avatar folder already
  exists, so preparation is only requested when that folder is absent.

The official code uses paths relative to the repository root (./models,
./results, ./musetalk/utils/dwpose/...), so the process chdirs there first.
Everything here requires an NVIDIA GPU with CUDA.
"""

import argparse
import hashlib
import importlib
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import wave
from pathlib import Path

logger = logging.getLogger("musetalk.engine")

AVATAR_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
FPS = 25  # MuseTalk was trained on 25 fps video; the real-time script uses --fps 25.


class EngineError(RuntimeError):
    """Model or avatar problem on the server side."""


class InvalidRequestError(ValueError):
    """Bad input from the caller (audio format, avatar id, ...)."""


def official_args(musetalk_dir: Path, batch_size: int, gpu_id: int) -> argparse.Namespace:
    """Same values as scripts/realtime_inference.py's argparse defaults + `inference.sh v1.5 realtime`."""
    return argparse.Namespace(
        version="v15",
        ffmpeg_path=shutil.which("ffmpeg") or "ffmpeg",
        gpu_id=gpu_id,
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        unet_model_path="./models/musetalkV15/unet.pth",
        whisper_dir="./models/whisper",
        inference_config="configs/inference/realtime.yaml",
        bbox_shift=0,
        result_dir="./results",
        extra_margin=10,
        fps=FPS,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
        batch_size=batch_size,
        output_vid_name=None,
        use_saved_coord=False,
        saved_coord=False,
        parsing_mode="jaw",
        left_cheek_width=90,
        right_cheek_width=90,
        skip_save_images=False,  # must be False: the script only writes the MP4 when images are saved
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_ffmpeg(arguments: list[str]) -> None:
    result = subprocess.run(["ffmpeg", "-y", "-v", "error", *arguments], capture_output=True, text=True)
    if result.returncode != 0:
        raise EngineError(f"ffmpeg failed: {result.stderr.strip()[-500:]}")


def validate_wav(path: Path, max_seconds: float) -> float:
    """Return the duration of a PCM WAV file or raise InvalidRequestError."""
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames, rate = wav_file.getnframes(), wav_file.getframerate()
            if wav_file.getnchannels() < 1 or rate <= 0:
                raise InvalidRequestError("WAV file has no audio channels.")
    except (wave.Error, EOFError) as error:
        raise InvalidRequestError(f"Audio must be a PCM WAV file: {error}") from error
    duration = frames / rate
    if duration <= 0.1:
        raise InvalidRequestError("Audio is empty or shorter than 0.1 seconds.")
    if duration > max_seconds:
        raise InvalidRequestError(f"Audio is {duration:.1f} s; the limit is {max_seconds:.0f} s.")
    return duration


class MuseTalkEngine:
    def __init__(
        self,
        musetalk_dir: Path,
        avatars_dir: Path,
        output_dir: Path,
        batch_size: int = 20,
        gpu_id: int = 0,
    ):
        self.musetalk_dir = musetalk_dir.resolve()
        self.avatars_dir = avatars_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self._rt = None  # the imported scripts.realtime_inference module
        self._avatars = {}  # avatar_id -> official Avatar instance
        self._lock = threading.Lock()  # one GPU job at a time; Avatar uses shared tmp dirs
        self.gpu_info: dict = {}

    # ---------- loading ----------

    def load(self) -> None:
        import torch

        # Most fundamental requirement first, so a CPU-only machine gets the right message.
        if not torch.cuda.is_available():
            raise EngineError("CUDA is not available. MuseTalk needs an NVIDIA GPU (see musetalk/README.md).")
        if not (self.musetalk_dir / "scripts" / "realtime_inference.py").is_file():
            raise EngineError(f"MuseTalk repository not found at {self.musetalk_dir}")
        missing = [
            path for path in (
                "models/musetalkV15/unet.pth", "models/musetalkV15/musetalk.json", "models/sd-vae/config.json",
                "models/whisper/config.json", "models/dwpose/dw-ll_ucoco_384.pth",
                "models/face-parse-bisent/79999_iter.pth", "models/face-parse-bisent/resnet18-5c106cde.pth",
            ) if not (self.musetalk_dir / path).is_file()
        ]
        if missing:
            raise EngineError(f"MuseTalk weights missing (run download_weights.sh): {', '.join(missing)}")

        self.gpu_info = {
            "name": torch.cuda.get_device_name(self.gpu_id),
            "memory_total_mb": round(torch.cuda.get_device_properties(self.gpu_id).total_memory / 2**20),
        }

        # The official code resolves weights/configs relative to the repo root.
        os.chdir(self.musetalk_dir)
        if str(self.musetalk_dir) not in sys.path:
            sys.path.insert(0, str(self.musetalk_dir))
        rt = importlib.import_module("scripts.realtime_inference")  # also loads DWPose + face detector

        args = official_args(self.musetalk_dir, self.batch_size, self.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        vae, unet, pe = rt.load_all_model(
            unet_model_path=args.unet_model_path,
            vae_type=args.vae_type,
            unet_config=args.unet_config,
            device=device,
        )
        timesteps = torch.tensor([0], device=device)
        pe = pe.half().to(device)
        vae.vae = vae.vae.half().to(device)
        unet.model = unet.model.half().to(device)

        audio_processor = rt.AudioProcessor(feature_extractor_path=args.whisper_dir)
        weight_dtype = unet.model.dtype
        whisper = rt.WhisperModel.from_pretrained(args.whisper_dir)
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)
        fp = rt.FaceParsing(left_cheek_width=args.left_cheek_width, right_cheek_width=args.right_cheek_width)

        for name, value in {
            "args": args, "device": device, "vae": vae, "unet": unet, "pe": pe, "timesteps": timesteps,
            "audio_processor": audio_processor, "weight_dtype": weight_dtype, "whisper": whisper, "fp": fp,
        }.items():
            setattr(rt, name, value)
        self._rt = rt
        logger.info("MuseTalk 1.5 models loaded on %s", self.gpu_info["name"])

    @property
    def loaded(self) -> bool:
        return self._rt is not None

    # ---------- avatars ----------

    def avatar_source(self, avatar_id: str) -> Path:
        """avatars/<id>/reference.<mp4|mov|mkv|avi|png|jpg|jpeg>; a video is preferred."""
        if not AVATAR_ID_PATTERN.match(avatar_id):
            raise InvalidRequestError(f"Invalid avatar id '{avatar_id}'.")
        folder = self.avatars_dir / avatar_id
        for extension in VIDEO_EXTENSIONS + IMAGE_EXTENSIONS:
            candidate = folder / f"reference{extension}"
            if candidate.is_file():
                return candidate
        raise InvalidRequestError(
            f"Avatar '{avatar_id}' has no reference file. Add {folder}/reference.mp4 (recommended) or reference.png."
        )

    def list_avatars(self) -> list[dict]:
        found = []
        if self.avatars_dir.is_dir():
            for folder in sorted(self.avatars_dir.iterdir()):
                if folder.is_dir() and AVATAR_ID_PATTERN.match(folder.name):
                    try:
                        source = self.avatar_source(folder.name).name
                    except InvalidRequestError:
                        source = None
                    found.append({"id": folder.name, "reference": source, "prepared": folder.name in self._avatars})
        return found

    def _prepared_input(self, avatar_id: str, source: Path) -> Path:
        """Normalise the reference into what the official Avatar accepts.

        Video -> re-encoded at 25 fps (MuseTalk's training frame rate).
        Image -> a one-frame directory of PNGs (the real-time script reads a
        video file or a directory of .png files).
        """
        work = self.musetalk_dir / "results" / "stella_sources" / avatar_id
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        if source.suffix.lower() in VIDEO_EXTENSIONS:
            target = work / "reference_25fps.mp4"
            _run_ffmpeg(["-i", str(source), "-vf", f"fps={FPS}", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target)])
            return target
        frames = work / "frames"
        frames.mkdir()
        _run_ffmpeg(["-i", str(source), "-frames:v", "1", str(frames / "00000000.png")])
        return frames

    def _avatar(self, avatar_id: str):
        """Return a ready official Avatar, preparing it on first use or when the reference changed."""
        if avatar_id in self._avatars:
            return self._avatars[avatar_id]
        source = self.avatar_source(avatar_id)
        fingerprint = _file_sha256(source)
        avatar_dir = self.musetalk_dir / "results" / "v15" / "avatars" / avatar_id
        marker = avatar_dir / "stella_reference.sha256"
        if avatar_dir.exists() and (not marker.is_file() or marker.read_text().strip() != fingerprint):
            logger.info("Reference for avatar %s changed; re-preparing", avatar_id)
            shutil.rmtree(avatar_dir)
        preparation = not avatar_dir.exists()
        video_path = str(self._prepared_input(avatar_id, source)) if preparation else str(source)

        started_at = time.perf_counter()
        logger.info("%s avatar %s from %s", "Preparing" if preparation else "Loading", avatar_id, source.name)
        avatar = self._rt.Avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=0,  # v1.5 always uses 0 (see official scripts)
            batch_size=self.batch_size,
            preparation=preparation,
        )
        if preparation:
            marker.write_text(fingerprint)
        logger.info("Avatar %s ready in %.1f s", avatar_id, time.perf_counter() - started_at)
        self._avatars[avatar_id] = avatar
        return avatar

    def prepare(self, avatar_id: str) -> float:
        started_at = time.perf_counter()
        with self._lock:
            self._avatar(avatar_id)
        return time.perf_counter() - started_at

    # ---------- generation ----------

    def generate(self, audio_path: Path, avatar_id: str) -> dict:
        """Render a lip-synced MP4 for a WAV file. Returns metadata including the output path."""
        if not self.loaded:
            raise EngineError("MuseTalk models are not loaded.")
        video_id = uuid.uuid4().hex
        self.output_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.output_dir / f"{video_id}.mp4"

        with self._lock:
            started_at = time.perf_counter()
            avatar = self._avatar(avatar_id)
            prepare_seconds = time.perf_counter() - started_at

            render_started_at = time.perf_counter()
            # Official call: writes <avatar_path>/vid_output/<name>.mp4 with the audio muxed in.
            avatar.inference(str(audio_path), video_id, FPS, False)
            raw_path = Path(avatar.video_out_path) / f"{video_id}.mp4"
            render_seconds = time.perf_counter() - render_started_at

        if not raw_path.is_file() or raw_path.stat().st_size == 0:
            raise EngineError("MuseTalk did not produce a video (check the service log for ffmpeg errors).")
        try:
            # Move the MP4 index to the front so browsers can start playback while downloading.
            _run_ffmpeg(["-i", str(raw_path), "-c", "copy", "-movflags", "+faststart", str(final_path)])
        finally:
            raw_path.unlink(missing_ok=True)

        return {
            "video_id": video_id,
            "path": final_path,
            "avatar_prepare_seconds": round(prepare_seconds, 3),
            "render_seconds": round(render_seconds, 3),
        }
