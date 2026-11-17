// AI Lab: test each model on its own through the orchestrator's existing per-model endpoints.
//   Whisper  POST /api/whisper/transcribe   (multipart file)
//   Qwen3    POST /api/qwen/generate        (JSON prompt / system_prompt / max_new_tokens, no history)
//   Kokoro   POST /api/kokoro/synthesize    (JSON text / voice / speed) -> WAV at /api/audio/<name>
//   MuseTalk POST /api/avatar/generate      (multipart WAV + avatar_id) -> MP4 at /api/avatar/videos/<id>
// Each run appends its own result card, so several runs (also in different tabs) never overwrite each other.
// Independent of app.js: shares no state with the conversation.
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const VIEWS = ["conversation", "lab"];
  const PANELS = ["whisper", "qwen", "kokoro", "musetalk"];
  const lab = { panel: "whisper", kokoroLatest: null, museAudio: null };

  // ---------- navigation (#conversation, #lab, #lab/qwen ...) ----------

  function route() {
    const [view, panel] = (location.hash.replace("#", "") || "conversation").split("/");
    const name = VIEWS.includes(view) ? view : "conversation";
    document.body.dataset.view = name;
    for (const v of VIEWS) $(`view-${v}`).hidden = v !== name;
    for (const tab of document.querySelectorAll(".tab")) {
      if (tab.dataset.view === name) tab.setAttribute("aria-current", "page");
      else tab.removeAttribute("aria-current");
    }
    $("status").hidden = name !== "conversation"; // conversation phase pill
    if (name === "lab") {
      showPanel(PANELS.includes(panel) ? panel : lab.panel);
      refreshStatus();
    }
  }

  function showPanel(panel) {
    lab.panel = panel;
    for (const section of document.querySelectorAll(".panel")) section.hidden = section.dataset.panel !== panel;
    for (const tab of document.querySelectorAll(".subtab")) tab.setAttribute("aria-selected", String(tab.dataset.panel === panel));
  }

  // ---------- service status (GET /health) ----------

  const STATUS_TEXT = {
    ready: "ready", loading: "loading model…", pending: "starting…", unreachable: "not running",
    unavailable: "unavailable", error: "error", disabled: "disabled", unauthorized: "API key rejected",
    preparing_avatar: "preparing avatar…",
  };

  function setServiceBadge(name, info) {
    const badge = document.querySelector(`.badge[data-svc="${name}"]`);
    const status = (info && info.status) || "unavailable";
    badge.dataset.state = status === "ready" ? "ready" : status === "disabled" ? "disabled" : "unavailable";
    const model = info && info.model ? ` · ${info.model}` : "";
    badge.textContent = `${STATUS_TEXT[status] || status}${status === "ready" ? model : ""}`;
    badge.title = (info && (info.error || info.detail)) || "";
  }

  async function refreshStatus() {
    for (const name of PANELS) {
      const badge = document.querySelector(`.badge[data-svc="${name}"]`);
      badge.dataset.state = "checking";
      badge.textContent = "checking…";
    }
    try {
      const res = await fetch("/health", { cache: "no-store" });
      const data = await res.json();
      for (const name of ["whisper", "qwen", "kokoro"]) setServiceBadge(name, (data.services || {})[name]);
      const avatar = data.avatar || {};
      setServiceBadge("musetalk", avatar);
      const notice = $("musetalk-notice");
      if (avatar.status === "disabled") {
        notice.textContent = "MuseTalk is disabled on this machine (MUSETALK_ENABLED=false). It runs on the NVIDIA GPU host; see musetalk/README.md. Requests here will report this status.";
      } else if (avatar.status !== "ready") {
        notice.textContent = `MuseTalk is not ready: ${avatar.error || avatar.status || "unknown"}.`;
      }
      notice.hidden = avatar.status === "ready";
    } catch {
      for (const name of PANELS) setServiceBadge(name, { status: "unreachable", error: "Could not read /health" });
    }
  }

  // ---------- requests and result cards ----------

  function errorDetail(data, res) {
    const d = data && (data.error || data.detail);
    if (Array.isArray(d)) return d.map((x) => `${(x.loc || []).slice(1).join(".") || "input"}: ${x.msg}`).join("; ");
    if (d) return typeof d === "string" ? d : JSON.stringify(d);
    return `${res.status} ${res.statusText}`;
  }

  async function call(url, options) {
    const startedAt = performance.now();
    let res;
    try {
      res = await fetch(url, options);
    } catch {
      return { ok: false, status: 0, data: null, ms: performance.now() - startedAt, error: "Could not reach the orchestrator. Is `docker compose up -d` running?" };
    }
    let data = null;
    try { data = await res.json(); } catch { /* non-JSON error body */ }
    const ms = performance.now() - startedAt;
    return { ok: res.ok, status: res.status, data, ms, error: res.ok ? null : errorDetail(data, res) };
  }

  const secs = (s) => (typeof s === "number" ? `${s.toFixed(2)} s` : "–");
  const kb = (bytes) => (bytes >= 1048576 ? `${(bytes / 1048576).toFixed(2)} MB` : `${(bytes / 1024).toFixed(0)} KB`);

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function metricsList(pairs) {
    const dl = el("dl", "result-metrics");
    for (const [label, value] of pairs) {
      if (value === undefined || value === null || value === "") continue;
      const div = el("div");
      div.append(el("dt", "", label), el("dd", "", String(value)));
      dl.append(div);
    }
    return dl;
  }

  /** A result card at the top of a panel's results; returns helpers to fill it. */
  function newResult(panel, title) {
    const card = el("article", "result pending");
    const head = el("div", "result-head");
    head.append(el("strong", "", title), el("span", "result-time", new Date().toLocaleTimeString()));
    const body = el("div", "result-body", "Running…");
    card.append(head, body);
    $(`${panel}-results`).prepend(card);
    return {
      done(nodes, metrics) {
        card.className = "result";
        body.replaceChildren(...nodes, metricsList(metrics));
      },
      fail(message, metrics = []) {
        card.className = "result failed";
        body.replaceChildren(el("p", "result-error", message), metricsList(metrics));
      },
    };
  }

  function block(label, text) {
    const wrap = el("div", "result-block");
    wrap.append(el("span", "result-label", label), el("p", "result-text", text));
    return wrap;
  }

  // ---------- Whisper ----------

  $("whisper-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = event.target.file.files[0];
    if (!file) return;
    const card = newResult("whisper", file.name);
    const form = new FormData();
    form.append("file", file, file.name);
    const r = await call("/api/whisper/transcribe", { method: "POST", body: form });
    const t = (r.data && r.data.timings) || {};
    const metrics = [["Whisper", secs(t.transcription_seconds)], ["Server total", secs(t.total_seconds)],
      ["Round trip", secs(r.ms / 1000)], ["File", `${kb(file.size)} ${file.type || ""}`.trim()]];
    if (!r.ok) return card.fail(`Whisper failed (${r.status || "network"}): ${r.error}`, metrics);
    card.done([block("Transcript", r.data.transcript || "(no speech detected)")], metrics);
  });

  // ---------- Qwen3 ----------

  $("qwen-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const f = event.target;
    const prompt = f.prompt.value.trim();
    if (!prompt) return;
    const body = { prompt };
    if (f.system_prompt.value.trim()) body.system_prompt = f.system_prompt.value.trim();
    if (f.max_new_tokens.value) body.max_new_tokens = Number(f.max_new_tokens.value);
    const card = newResult("qwen", prompt.length > 60 ? `${prompt.slice(0, 60)}…` : prompt);
    const r = await call("/api/qwen/generate", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const t = (r.data && r.data.timings) || {};
    const tokens = r.data && r.data.generated_tokens;
    const metrics = [["Qwen3 generation", secs(t.generation_seconds)], ["Tokens", tokens],
      ["Tokens / s", tokens && t.generation_seconds ? (tokens / t.generation_seconds).toFixed(1) : null],
      ["Max new tokens", body.max_new_tokens || "default"], ["Server total", secs(t.total_seconds)],
      ["Round trip", secs(r.ms / 1000)]];
    if (!r.ok) return card.fail(`Qwen3 failed (${r.status || "network"}): ${r.error}`, metrics);
    const nodes = [block("Prompt", prompt)];
    if (body.system_prompt) nodes.push(block("System prompt", body.system_prompt));
    nodes.push(block("Response", r.data.response));
    card.done(nodes, metrics);
  });

  // ---------- Kokoro ----------

  function wavInfo(buffer) {
    const v = new DataView(buffer);
    const tag = (o) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
    if (buffer.byteLength < 44 || tag(0) !== "RIFF" || tag(8) !== "WAVE") return null;
    return { channels: v.getUint16(22, true), rate: v.getUint32(24, true), bits: v.getUint16(34, true) };
  }

  $("kokoro-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const f = event.target;
    const text = f.text.value.trim();
    if (!text) return;
    const body = { text, voice: f.voice.value, speed: Number(f.speed.value) || 1.0 };
    const card = newResult("kokoro", `${body.voice} · ${text.length > 50 ? `${text.slice(0, 50)}…` : text}`);
    const r = await call("/api/kokoro/synthesize", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const t = (r.data && r.data.timings) || {};
    const metrics = [["Kokoro synthesis", secs(t.synthesis_seconds)], ["Audio length", secs(r.data && r.data.audio_duration_seconds)],
      ["Voice", body.voice], ["Speed", body.speed], ["Server total", secs(t.total_seconds)], ["Round trip", secs(r.ms / 1000)]];
    if (!r.ok) return card.fail(`Kokoro failed (${r.status || "network"}): ${r.error}`, metrics);

    let blob;
    try {
      const audioRes = await fetch(r.data.audio_url);
      if (!audioRes.ok) throw new Error(`HTTP ${audioRes.status}`);
      blob = await audioRes.blob();
    } catch (error) {
      return card.fail(`Audio was generated but could not be downloaded: ${error.message}`, metrics);
    }
    const info = wavInfo(await blob.arrayBuffer());
    if (info) metrics.push(["Format", `WAV ${info.rate / 1000} kHz · ${info.channels === 1 ? "mono" : `${info.channels} ch`} · ${info.bits}-bit`]);
    metrics.push(["Size", kb(blob.size)], ["Saved as", r.data.audio_file]);

    const player = el("audio", "lab-audio");
    player.controls = true;
    player.src = URL.createObjectURL(blob);
    const download = el("a", "hint", "Download WAV");
    download.href = r.data.audio_url;
    download.download = "";
    card.done([block("Text", text), player, download], metrics);
    player.play().catch(() => { /* autoplay blocked: controls are visible */ });

    lab.kokoroLatest = { blob, name: r.data.audio_url.split("/").pop() };
    $("use-kokoro").disabled = false;
  });

  // ---------- MuseTalk ----------

  async function loadAvatars() {
    const select = $("lab-avatar");
    try {
      const res = await fetch("/api/avatar/avatars", { cache: "no-store" });
      const data = await res.json();
      select.replaceChildren();
      for (const a of data.avatars || []) {
        const label = `${a.id}${a.default ? " (default)" : ""} · ${a.reference || "no reference file yet"}`;
        const option = el("option", "", label);
        option.value = a.id;
        option.selected = a.default;
        select.append(option);
      }
      if (!select.options.length) select.append(Object.assign(el("option", "", "No avatars in avatars/"), { value: "" }));
    } catch {
      select.replaceChildren(Object.assign(el("option", "", "Could not load avatars"), { value: "" }));
    }
  }

  function setMuseSource(source) {
    lab.museAudio = source;
    $("musetalk-source").textContent = source ? `Using ${source.name} (${kb(source.blob.size)})` : "No audio selected.";
  }

  $("musetalk-form").file.addEventListener("change", (event) => {
    const file = event.target.files[0];
    setMuseSource(file ? { blob: file, name: file.name } : null);
  });

  $("use-kokoro").addEventListener("click", () => {
    if (!lab.kokoroLatest) return;
    $("musetalk-form").file.value = "";
    setMuseSource(lab.kokoroLatest);
  });

  $("musetalk-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const source = lab.museAudio;
    if (!source) {
      $("musetalk-source").textContent = "Choose a WAV file or use the latest Kokoro audio first.";
      return;
    }
    const avatarId = event.target.avatar_id.value;
    const card = newResult("musetalk", `${avatarId || "default avatar"} · ${source.name}`);
    const form = new FormData();
    form.append("file", source.blob, source.name);
    if (avatarId) form.append("avatar_id", avatarId);
    const r = await call("/api/avatar/generate", { method: "POST", body: form });
    const d = r.data || {};
    const t = d.timings || {};
    const metrics = [["MuseTalk render", secs(t.musetalk_render_seconds)], ["MuseTalk service", secs(t.musetalk_service_seconds)],
      ["Avatar total", secs(t.avatar_seconds)], ["Audio length", secs(d.audio_duration_seconds)],
      ["Video size", d.video_bytes ? kb(d.video_bytes) : null], ["Round trip", secs(r.ms / 1000)]];
    if (!r.ok) {
      const label = d.status ? `MuseTalk ${d.status}` : `MuseTalk request failed (${r.status || "network"})`;
      return card.fail(`${label}: ${r.error}`, metrics);
    }
    const video = el("video", "lab-video");
    video.controls = true;
    video.playsInline = true;
    video.src = d.video_url;
    video.onerror = () => card.fail(`The browser could not play ${d.video_url}.`, metrics);
    card.done([video], metrics);
    video.play().catch(() => { /* autoplay blocked: controls are visible */ });
  });

  // ---------- wiring ----------

  for (const tab of document.querySelectorAll(".subtab")) {
    tab.addEventListener("click", () => { history.replaceState(null, "", `#lab/${tab.dataset.panel}`); showPanel(tab.dataset.panel); });
  }
  $("lab-refresh").addEventListener("click", () => { refreshStatus(); loadAvatars(); });
  // Same voices as the conversation selector.
  $("lab-voice").replaceChildren(...[...$("voice").options].map((o) => new Option(o.textContent, o.value)));
  window.addEventListener("hashchange", route);
  loadAvatars();
  route();
})();
