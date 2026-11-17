// Stella: browser microphone -> orchestrator /api/conversation/turn -> avatar video (or voice) reply.
// No framework: one state machine (idle | recording | processing | speaking | error).
//
// A/V sync: when MuseTalk returns a video, that MP4 already contains Kokoro's audio
// (muxed by MuseTalk), so the page plays ONLY the video, never the video plus a
// separate audio element. Without a video it falls back to the Kokoro WAV.
"use strict";

const ENDPOINT = "/api/conversation/turn";
const STORAGE_KEY = "stella.conversation.v1";
const AVATAR_PREF_KEY = "stella.avatar.enabled";
const MAX_RECORD_SECONDS = 60;
const MIN_RECORD_MS = 500;
// First type the browser supports wins; Chrome/Edge: webm, Firefox: ogg, Safari: mp4.
const MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4", "audio/webm", "audio/ogg"];

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"), conversation: $("conversation"), empty: $("empty"), message: $("message"),
  mic: $("mic"), micLabel: $("mic-label"), timer: $("timer"), stopAudio: $("stop-audio"),
  replay: $("replay"), voice: $("voice"), clear: $("clear"), metrics: $("metrics"), player: $("player"),
  stage: $("stage"), idleImg: $("avatar-idle"), placeholder: $("avatar-placeholder"), video: $("avatar-video"),
  overlay: $("stage-overlay"), overlayText: $("overlay-text"), tapToPlay: $("tap-to-play"),
  badge: $("avatar-badge"), avatarToggle: $("avatar-toggle"),
};

const state = {
  phase: "idle",
  turns: loadTurns(),          // [{ user, assistant, timings, avatar }]
  stream: null,
  recorder: null,
  chunks: [],
  recordStartedAt: 0,
  stopClickedAt: 0,
  timerHandle: null,
  audioUrl: null,
  media: null,                 // { kind: "video" | "audio", el, turn } - the reply currently loaded
  unlocked: false,
  avatarEnabled: false,        // server has MUSETALK_ENABLED=true
};

// ---------- state & rendering ----------

const PHASE_LABEL = { idle: "Idle", recording: "Recording", processing: "Processing", speaking: "Speaking", error: "Error" };

function setPhase(phase) {
  state.phase = phase;
  els.status.dataset.phase = phase;
  els.status.textContent = PHASE_LABEL[phase];
  const recording = phase === "recording";
  els.mic.classList.toggle("recording", recording);
  els.mic.disabled = phase === "processing" || phase === "speaking";
  els.micLabel.textContent = recording ? "Stop" : "Start";
  els.mic.setAttribute("aria-label", recording ? "Stop recording and send" : "Start recording");
  els.stopAudio.disabled = phase !== "speaking";
  els.replay.disabled = !state.media || phase === "processing" || recording || phase === "speaking";
  els.clear.disabled = phase === "processing" || recording;
  els.voice.disabled = phase === "processing" || recording;
  els.avatarToggle.disabled = !state.avatarEnabled || phase === "processing" || recording;
  els.timer.hidden = !recording;
  els.overlay.hidden = phase !== "processing";
  els.stage.dataset.mode = phase === "speaking" ? "speaking" : "idle";
}

function showMessage(text, kind = "error") {
  els.message.textContent = text;
  els.message.className = kind === "info" ? "message info" : "message";
  els.message.hidden = !text;
}

function fail(text) {
  showMessage(text);
  setPhase("error");
}

function bubble(role, text, meta, pending = false) {
  const div = document.createElement("div");
  div.className = `bubble ${role}${pending ? " pending" : ""}`;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = role === "user" ? "You" : "Stella";
  div.append(who, document.createTextNode(text));
  if (meta) {
    const m = document.createElement("span");
    m.className = "meta";
    m.textContent = meta;
    div.append(m);
  }
  return div;
}

function renderConversation(pendingText) {
  els.conversation.replaceChildren();
  if (!state.turns.length && !pendingText) els.conversation.append(els.empty);
  for (const turn of state.turns) {
    els.conversation.append(bubble("user", turn.user));
    els.conversation.append(bubble("assistant", turn.assistant, turnMeta(turn)));
  }
  if (pendingText) els.conversation.append(bubble("assistant", pendingText, null, true));
  els.conversation.scrollTop = els.conversation.scrollHeight;
}

const secs = (s) => (typeof s === "number" ? `${s.toFixed(2)} s` : "–");
const ms = (m) => (typeof m === "number" ? `${(m / 1000).toFixed(2)} s` : "–");

function turnMeta(turn) {
  const t = turn.timings;
  if (!t) return "";
  const parts = [`server ${secs(t.server)}`];
  if (typeof t.stoptoplay === "number") parts.push(`stop→playback ${ms(t.stoptoplay)}`);
  if (turn.avatarStatus && turn.avatarStatus !== "ready") parts.push(`avatar ${turn.avatarStatus}`);
  else if (turn.avatarStatus === "ready") parts.push("avatar video");
  return parts.join(" · ");
}

function renderMetrics(t = {}) {
  const values = {
    upload: ms(t.upload), transcription: secs(t.transcription), generation: secs(t.generation),
    synthesis: secs(t.synthesis), musetalk: secs(t.musetalk), avatar: secs(t.avatar),
    server: secs(t.server), roundtrip: ms(t.roundtrip), stoptoplay: ms(t.stoptoplay), recording: t.recording || "–",
  };
  for (const dd of els.metrics.querySelectorAll("dd")) dd.textContent = values[dd.dataset.k] ?? "–";
}

// ---------- avatar status ----------

const AVATAR_LABEL = {
  ready: "Avatar: ready", disabled: "Avatar: disabled (voice only)", off: "Avatar: off (voice only)",
  unavailable: "Avatar: unavailable (voice only)", unreachable: "Avatar: GPU service unreachable (voice only)",
  unauthorized: "Avatar: API key rejected (voice only)", loading: "Avatar: GPU service loading…",
  preparing_avatar: "Avatar: preparing Stella…", failed: "Avatar: last video failed", error: "Avatar: GPU service error",
};

function setBadge(status, detail) {
  const key = AVATAR_LABEL[status] ? status : "unavailable";
  els.badge.dataset.state = key === "ready" ? "ready" : (key === "disabled" || key === "off" ? "disabled" : "unavailable");
  els.badge.textContent = AVATAR_LABEL[key];
  els.badge.title = detail || "";
}

async function loadAvatarStatus() {
  try {
    const res = await fetch("/api/avatar/status", { cache: "no-store" });
    const data = await res.json();
    state.avatarEnabled = !!data.enabled;
    if (data.idle_image_url) {
      els.idleImg.src = `${data.idle_image_url}?t=${Date.now()}`;
      els.idleImg.onload = () => { els.idleImg.hidden = false; els.placeholder.hidden = true; };
    }
    if (!data.enabled) setBadge("disabled");
    else if (!els.avatarToggle.checked) setBadge("off");
    else setBadge(data.service && data.service.status, data.service && data.service.error);
  } catch {
    setBadge("unavailable", "Could not read /api/avatar/status");
  }
  setPhase(state.phase);
}

function loadAvatarPreference() {
  try { els.avatarToggle.checked = localStorage.getItem(AVATAR_PREF_KEY) !== "false"; } catch { /* storage unavailable */ }
}

// ---------- session history (sessionStorage: survives reloads, not new tabs) ----------

function loadTurns() {
  try { return JSON.parse(sessionStorage.getItem(STORAGE_KEY)) || []; } catch { return []; }
}
function saveTurns() {
  try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state.turns)); } catch { /* storage unavailable */ }
}
function historyForServer() {
  return state.turns.flatMap((t) => [{ role: "user", content: t.user }, { role: "assistant", content: t.assistant }]);
}

// ---------- recording ----------

function pickMimeType() {
  if (typeof MediaRecorder.isTypeSupported !== "function") return "";
  return MIME_CANDIDATES.find((t) => MediaRecorder.isTypeSupported(t)) || "";
}

function extensionFor(mime) {
  if (mime.includes("ogg")) return "ogg";
  if (mime.includes("mp4") || mime.includes("aac")) return "m4a";
  if (mime.includes("wav")) return "wav";
  return "webm";
}

function micErrorMessage(error) {
  switch (error && error.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Microphone permission was denied. Allow it via the icon at the left of the address bar, then try again.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No microphone was found. Connect one or check Windows sound settings.";
    case "NotReadableError":
      return "The microphone is in use by another application or could not be started.";
    default:
      return `Could not access the microphone: ${error && error.message ? error.message : error}`;
  }
}

async function startRecording() {
  if (state.phase === "processing" || state.phase === "speaking") return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return fail("This browser does not allow microphone access here. Use a current Chrome, Edge or Firefox at http://localhost:8000.");
  }
  if (typeof window.MediaRecorder === "undefined") {
    return fail("This browser does not support audio recording (MediaRecorder).");
  }
  unlockMedia();
  stopPlayback();
  showMessage("");
  els.tapToPlay.hidden = true;

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (error) {
    return fail(micErrorMessage(error));
  }

  const mimeType = pickMimeType();
  try {
    state.recorder = mimeType ? new MediaRecorder(state.stream, { mimeType }) : new MediaRecorder(state.stream);
  } catch (error) {
    releaseMic();
    return fail(`Recording format not supported by this browser: ${error.message}`);
  }
  state.chunks = [];
  state.recorder.ondataavailable = (e) => { if (e.data && e.data.size) state.chunks.push(e.data); };
  state.recorder.onstop = onRecordingStopped;
  state.recorder.onerror = (e) => { releaseMic(); fail(`Recording failed: ${e.error ? e.error.message : "unknown error"}`); };
  state.recorder.start();
  state.recordStartedAt = performance.now();
  setPhase("recording");

  state.timerHandle = setInterval(() => {
    const elapsed = (performance.now() - state.recordStartedAt) / 1000;
    els.timer.textContent = `${elapsed.toFixed(1)} s`;
    if (elapsed >= MAX_RECORD_SECONDS) stopRecording();
  }, 100);
}

function stopRecording() {
  if (state.phase !== "recording" || !state.recorder) return;
  state.stopClickedAt = performance.now();
  clearInterval(state.timerHandle);
  if (state.recorder.state !== "inactive") state.recorder.stop(); // fires onstop after the final chunk
  releaseMic();
}

function releaseMic() {
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
}

async function onRecordingStopped() {
  const mime = (state.recorder && state.recorder.mimeType) || pickMimeType() || "audio/webm";
  const blob = new Blob(state.chunks, { type: mime });
  const recordedMs = state.stopClickedAt - state.recordStartedAt;
  state.recorder = null;

  if (recordedMs < MIN_RECORD_MS || blob.size < 1000) {
    return fail("That recording was too short. Hold the recording for at least a second while you speak.");
  }
  const recordingInfo = `${(recordedMs / 1000).toFixed(1)} s ${mime.split(";")[0].replace("audio/", "")} ${(blob.size / 1024).toFixed(0)} KB`;
  await sendTurn(blob, mime, recordingInfo);
}

// ---------- request ----------

function postForm(form) {
  // XHR rather than fetch: it reports when the upload finished, so upload time can be measured.
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const sentAt = performance.now();
    let uploadedAt = null;
    xhr.open("POST", ENDPOINT);
    xhr.responseType = "json";
    xhr.timeout = 10 * 60 * 1000;
    xhr.upload.onload = () => { uploadedAt = performance.now(); };
    xhr.onload = () => resolve({
      status: xhr.status,
      data: xhr.response,
      upload: uploadedAt === null ? null : uploadedAt - sentAt,
      roundtrip: performance.now() - sentAt,
    });
    xhr.onerror = () => reject(new Error("Could not reach the orchestrator. Is `docker compose up -d` running?"));
    xhr.ontimeout = () => reject(new Error("The request timed out."));
    xhr.send(form);
  });
}

function errorText(status, data) {
  const detail = (data && (data.error || data.detail)) || "";
  const text = typeof detail === "string" ? detail : JSON.stringify(detail);
  if (data && data.no_speech) return "I didn't catch any speech. Speak a little louder or closer to the microphone and try again.";
  if (status === 422 && data && data.failed_stage === "transcription") return `Could not read that audio: ${text}`;
  if (status === 429) return "Still working on the previous turn. Please wait for it to finish.";
  if (status === 503) return `A model service is not ready: ${text} Check http://localhost:8000/health.`;
  return `Request failed (${status}): ${text || "unknown error"}`;
}

async function sendTurn(blob, mime, recordingInfo) {
  setPhase("processing");
  const wantAvatar = state.avatarEnabled && els.avatarToggle.checked;
  els.overlayText.textContent = wantAvatar ? "Thinking and rendering Stella…" : "Thinking…";
  renderConversation("Listening and thinking…");

  const form = new FormData();
  form.append("file", blob, `recording.${extensionFor(mime)}`);
  form.append("history", JSON.stringify(historyForServer()));
  form.append("voice", els.voice.value);
  form.append("avatar", wantAvatar ? "true" : "false");

  let res;
  try {
    res = await postForm(form);
  } catch (error) {
    renderConversation();
    return fail(error.message);
  }

  const data = res.data || {};
  if (res.status !== 200) {
    renderConversation();
    if (data.transcript) showMessage(`Heard: "${data.transcript}". ${errorText(res.status, data)}`);
    else showMessage(errorText(res.status, data));
    return setPhase("error");
  }

  const t = data.timings || {};
  const av = data.avatar || { status: "disabled" };
  const avt = av.timings || {};
  const timings = {
    upload: res.upload, roundtrip: res.roundtrip, recording: recordingInfo,
    transcription: t.transcription_seconds, generation: t.generation_seconds,
    synthesis: t.synthesis_seconds, server: t.total_seconds,
    musetalk: avt.musetalk_render_seconds, avatar: avt.avatar_seconds,
  };
  const turn = { user: data.transcript, assistant: data.response, timings, avatarStatus: av.status };
  state.turns.push(turn);
  saveTurns();
  renderConversation();
  renderMetrics(timings);
  if (state.avatarEnabled) setBadge(av.status, av.error);

  if (av.status === "ready" && av.video_url) {
    loadVideo(av.video_url, turn, data);
  } else {
    if (av.status === "unavailable" || av.status === "failed") {
      showMessage(`Avatar video ${av.status}: ${av.error || "no details"}. Playing Stella's voice only.`, "info");
    }
    loadAudio(data.audio_base64, data.audio_mime || "audio/wav", turn);
  }
}

// ---------- playback (exactly one media element per reply) ----------

function base64ToBlob(b64, mime) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

// A tiny silent WAV played on the first click "unlocks" the <audio> and <video>
// elements, so replies can auto-play later even though they arrive seconds after
// the user gesture (needed by Safari; harmless elsewhere).
function silentWavUrl() {
  const samples = 800, buf = new ArrayBuffer(44 + samples * 2), v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + samples * 2, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 8000, true); v.setUint32(28, 16000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, samples * 2, true);
  return URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
}

function unlockMedia() {
  if (state.unlocked) return;
  state.unlocked = true;
  const url = silentWavUrl();
  for (const el of [els.player, els.video]) {
    el.src = url;
    el.play().catch(() => {}).finally(() => el.pause());
  }
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function showIdle() {
  els.video.hidden = true;
  els.video.removeAttribute("controls");
}

function loadVideo(url, turn, data) {
  els.player.pause();
  els.player.hidden = true;
  els.video.onerror = () => {
    // The MP4 could not be decoded/fetched: keep the conversation going with the voice.
    showIdle();
    showMessage("Stella's video could not be played; playing her voice only.", "info");
    setBadge("failed", "Browser could not play the video");
    loadAudio(data.audio_base64, data.audio_mime || "audio/wav", turn);
  };
  els.video.src = url;
  els.video.muted = false;
  state.media = { kind: "video", el: els.video, turn };
  startPlayback();
}

function loadAudio(b64, mime, turn) {
  if (!b64) return fail("The server returned no audio.");
  els.video.pause();
  showIdle();
  if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
  state.audioUrl = URL.createObjectURL(base64ToBlob(b64, mime));
  els.player.src = state.audioUrl;
  els.player.hidden = false;
  state.media = { kind: "audio", el: els.player, turn };
  startPlayback();
}

function startPlayback() {
  const media = state.media;
  if (!media) return;
  const { el, turn } = media;
  el.onplaying = () => {
    if (turn && typeof turn.timings.stoptoplay !== "number") {
      turn.timings.stoptoplay = performance.now() - state.stopClickedAt;
      saveTurns();
      renderMetrics(turn.timings);
      renderConversation();
    }
    els.tapToPlay.hidden = true;
    if (media.kind === "video") { els.video.hidden = false; els.video.setAttribute("controls", ""); }
    setPhase("speaking");
  };
  el.onended = () => {
    if (media.kind === "video") showIdle();
    setPhase("idle");
  };
  el.onpause = () => { if (state.phase === "speaking" && !el.ended) setPhase("idle"); };
  el.currentTime = 0;
  el.play().catch((error) => {
    // Autoplay blocked: keep the reply ready behind a button.
    els.tapToPlay.hidden = false;
    showMessage(`The browser blocked automatic playback (${error.name}). Press "Play Stella's reply".`, "info");
    setPhase("idle");
  });
}

function stopPlayback() {
  for (const el of [els.video, els.player]) {
    if (!el.paused) el.pause();
  }
  showIdle();
  if (state.phase === "speaking") setPhase("idle");
}

// ---------- wiring ----------

els.mic.addEventListener("click", () => (state.phase === "recording" ? stopRecording() : startRecording()));
els.stopAudio.addEventListener("click", stopPlayback);
els.replay.addEventListener("click", () => { showMessage(""); startPlayback(); });
els.tapToPlay.addEventListener("click", () => { showMessage(""); startPlayback(); });
els.avatarToggle.addEventListener("change", () => {
  try { localStorage.setItem(AVATAR_PREF_KEY, String(els.avatarToggle.checked)); } catch { /* ignore */ }
  loadAvatarStatus();
});
els.clear.addEventListener("click", () => {
  stopPlayback();
  state.turns = [];
  state.media = null;
  els.player.hidden = true;
  saveTurns();
  showMessage("");
  renderConversation();
  renderMetrics();
  setPhase("idle");
});
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat || e.target.closest("select, button, input, textarea, a")) return;
  if (document.body.dataset.view === "lab") return; // shortcut belongs to the Conversation view
  e.preventDefault();
  if (state.phase === "recording") stopRecording();
  else if (state.phase === "idle" || state.phase === "error") startRecording();
});

loadAvatarPreference();
renderConversation();
renderMetrics(state.turns.length ? state.turns[state.turns.length - 1].timings : {});
setPhase("idle");
loadAvatarStatus();
