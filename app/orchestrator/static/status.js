// System status panel (both views): GET /api/system/status every 10 s while visible.
//   - per service: container reachability, model readiness, device, load time, uptime, VRAM, errors
//   - GPU: utilization and memory (nvidia-smi via a model container), vLLM KV-cache and queue
//   - recent processing times per stage (orchestrator memory, last 20 runs)
// Read-only: it never calls a model's inference endpoint.
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const REFRESH_MS = 10000;
  const SERVICES = [
    ["whisper", "Whisper", "speech → text"],
    ["qwen", "Qwen3", "LLM"],
    ["kokoro", "Kokoro", "text → speech"],
    ["musetalk", "MuseTalk", "avatar video"],
  ];
  const STAGES = [
    ["whisper", "Whisper"], ["qwen", "Qwen3 generation"], ["qwen_ttft", "Qwen3 first token"],
    ["kokoro", "Kokoro"], ["musetalk", "MuseTalk render"], ["conversation_turn", "Stella Chat turn (server)"],
  ];
  const STATE_TEXT = {
    ready: "ready", loading: "loading model…", pending: "starting…", unreachable: "not reachable",
    error: "error", disabled: "disabled", unauthorized: "API key rejected", preparing_avatar: "preparing avatar…",
  };
  let inFlight = false;

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  };
  const secs = (s) => (typeof s === "number" ? `${s.toFixed(2)} s` : null);
  const gb = (mb) => (typeof mb === "number" ? `${(mb / 1024).toFixed(1)} GB` : null);
  const pct = (v) => (typeof v === "number" ? `${Math.round(v)} %` : null);
  function duration(seconds) {
    if (typeof seconds !== "number") return null;
    if (seconds < 90) return `${Math.round(seconds)} s`;
    if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
    return `${(seconds / 3600).toFixed(1)} h`;
  }

  function list(pairs) {
    const dl = el("dl", "svc-list");
    for (const [label, value] of pairs) {
      if (value === undefined || value === null || value === "") continue;
      const row = el("div");
      row.append(el("dt", "", label), el("dd", "", String(value)));
      dl.append(row);
    }
    return dl;
  }

  function stateOf(info) {
    if (!info) return "down";
    if (info.status === "ready") return "ready";
    if (info.status === "disabled") return "disabled";
    if (["loading", "pending", "preparing_avatar"].includes(info.status)) return "loading";
    return "down";
  }

  function containerText(info) {
    switch (info && info.container) {
      case "running": return "running (responds)";
      case "disabled": return "not used (MUSETALK_ENABLED=false)";
      case "not_reachable": return "not reachable: stopped, crashed or still starting";
      default: return "unknown";
    }
  }

  function vram(info) {
    if (typeof info.gpu_memory_reserved_mb !== "number") return null;
    const peak = typeof info.gpu_memory_peak_reserved_mb === "number" ? ` · peak ${gb(info.gpu_memory_peak_reserved_mb)}` : "";
    return `${gb(info.gpu_memory_reserved_mb)}${peak} (PyTorch)`;
  }

  function lastRun(perf, stage) {
    const p = perf && perf[stage];
    return p ? `${secs(p.last_seconds)} · avg ${secs(p.avg_seconds)} (${p.count})` : null;
  }

  function serviceCard(key, title, role, info, data) {
    const card = el("article", "svc");
    const state = stateOf(info);
    card.dataset.state = state;
    const head = el("header");
    const badge = el("span", "badge", STATE_TEXT[info && info.status] || (info && info.status) || "unknown");
    badge.dataset.state = state === "ready" ? "ready" : state === "disabled" ? "disabled" : "unavailable";
    head.append(el("strong", "", title), el("span", "hint", role), badge);
    card.append(head);

    const gpuTotal = data.gpu && data.gpu.available && data.gpu.gpus[0] ? data.gpu.gpus[0].memory_total_mb : null;
    const pairs = [["Container", containerText(info)], ["Model", info && info.model]];
    if (key === "qwen") {
      pairs.push(["Backend", info && info.backend ? info.backend : data.qwen_backend]);
      const engine = (info && info.engine) || {};
      if (typeof engine.kv_cache_usage === "number") pairs.push(["KV cache in use", pct(engine.kv_cache_usage * 100)]);
      if (typeof engine.requests_running === "number") {
        pairs.push(["Requests", `${engine.requests_running} running · ${engine.requests_waiting || 0} waiting`]);
      }
      const fraction = info && info.gpu_memory_budget_fraction;
      if (fraction) pairs.push(["VRAM budget", `${Math.round(fraction * 100)} % of GPU${gpuTotal ? ` ≈ ${gb(fraction * gpuTotal)}` : ""} (fixed)`]);
      pairs.push(["Last generation", lastRun(data.performance, "qwen")], ["Last first token", lastRun(data.performance, "qwen_ttft")]);
    } else if (key === "musetalk") {
      if (info && typeof info.cuda_available === "boolean") pairs.push(["CUDA", info.cuda_available ? "available" : "NOT available"]);
      if (info && info.gpu && info.gpu.name) pairs.push(["GPU", info.gpu.name]);
      pairs.push(["VRAM", info && vram(info)], ["Default avatar", info && info.default_avatar]);
      pairs.push(["Load time", info && secs(info.load_seconds)], ["Up for", info && duration(info.uptime_seconds)]);
      pairs.push(["Last render", lastRun(data.performance, "musetalk")]);
    } else {
      const device = info && info.device ? `${info.device}${info.gpu ? ` · ${info.gpu}` : ""}` : null;
      pairs.push(["Device", device], ["VRAM", info && vram(info)]);
      pairs.push(["Load time", info && secs(info.load_seconds)], ["Up for", info && duration(info.uptime_seconds)]);
      pairs.push(["Last run", lastRun(data.performance, key)]);
    }
    pairs.push(["Health check", info && typeof info.health_check_ms === "number" ? `${info.health_check_ms} ms` : null]);
    card.append(list(pairs));
    const error = info && (info.error || info.detail || info.gpu_error);
    if (error) card.append(el("p", "svc-error", error));
    return card;
  }

  function gpuCard(data) {
    const card = $("sys-gpu");
    const head = el("header");
    head.append(el("strong", "", "GPU"));
    const gpu = data.gpu || {};
    if (!gpu.available) {
      card.replaceChildren(head, el("p", "hint", `Not available: ${gpu.error || "no data"}. (Expected in the CPU laptop stack.)`));
      return;
    }
    const nodes = [head];
    for (const g of gpu.gpus) {
      const used = g.memory_used_mb, total = g.memory_total_mb;
      const share = typeof used === "number" && total ? (used / total) * 100 : null;
      const bar = el("div", "meter");
      const fill = el("span");
      fill.style.width = `${Math.min(100, share || 0)}%`;
      if (share >= 90) bar.dataset.level = "high";
      bar.append(fill);
      nodes.push(el("p", "gpu-name", `${g.name}${gpu.gpus.length > 1 ? ` (#${g.index})` : ""}`), bar, list([
        ["VRAM used (all containers)", used !== null ? `${gb(used)} / ${gb(total)}${share !== null ? ` · ${Math.round(share)} %` : ""}` : null],
        ["Utilization", pct(g.utilization_percent)], ["Temperature", typeof g.temperature_c === "number" ? `${g.temperature_c} °C` : null],
        ["Power", typeof g.power_w === "number" ? `${Math.round(g.power_w)} W` : null], ["Driver", g.driver_version],
      ]));
    }
    nodes.push(el("p", "hint", "Per-container VRAM is listed on each service card (PyTorch-reserved memory; vLLM uses its fixed budget)."));
    card.replaceChildren(...nodes);
  }

  function perfCard(data) {
    const card = $("sys-perf");
    const head = el("header");
    head.append(el("strong", "", "Recent processing times"), el("span", "hint", "last 20 runs, since orchestrator start"));
    const perf = data.performance || {};
    const rows = STAGES.filter(([key]) => perf[key]);
    if (!rows.length) {
      card.replaceChildren(head, el("p", "hint", "No runs yet. Use the AI Lab or Stella Chat."));
      return;
    }
    const table = el("table", "perf");
    const thead = el("tr");
    for (const h of ["Stage", "Last", "Avg", "Min", "Max", "Runs"]) thead.append(el("th", "", h));
    table.append(thead);
    for (const [key, label] of rows) {
      const p = perf[key];
      const tr = el("tr");
      for (const v of [label, secs(p.last_seconds), secs(p.avg_seconds), secs(p.min_seconds), secs(p.max_seconds), p.count]) tr.append(el("td", "", v));
      table.append(tr);
    }
    card.replaceChildren(head, table, el("p", "hint", "Browser-side latency (upload, stop → playback) is shown under Stella Chat → Last turn."));
  }

  function render(data) {
    const services = data.services || {};
    const grid = $("sys-services");
    grid.replaceChildren(...SERVICES.map(([key, title, role]) => serviceCard(key, title, role, services[key], data)));
    gpuCard(data);
    perfCard(data);
    const needed = SERVICES.filter(([key]) => stateOf(services[key]) !== "disabled");
    const ready = needed.filter(([key]) => stateOf(services[key]) === "ready").length;
    const g = data.gpu && data.gpu.available ? data.gpu.gpus[0] : null;
    const gpuText = g ? ` · GPU ${pct(g.utilization_percent) || "?"} · ${gb(g.memory_used_mb)} / ${gb(g.memory_total_mb)}` : "";
    const summary = $("sys-summary");
    summary.textContent = `${ready}/${needed.length} services ready${gpuText}`;
    summary.dataset.state = ready === needed.length ? "ready" : "unavailable";
    $("sys-updated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
  }

  async function refresh() {
    if (inFlight) return;
    inFlight = true;
    try {
      const res = await fetch("/api/system/status", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      render(await res.json());
    } catch (error) {
      $("sys-summary").textContent = "orchestrator not reachable";
      $("sys-summary").dataset.state = "unavailable";
      $("sys-updated").textContent = `Could not read /api/system/status (${error.message}). Is the orchestrator container running?`;
    } finally {
      inFlight = false;
    }
  }

  // Browsers expose the microphone only in a secure context: https:// or http://localhost.
  function checkSecureContext() {
    if (window.isSecureContext) return;
    const warning = $("secure-warning");
    warning.textContent = `Microphone recording is blocked: this page was opened at ${location.origin}, which the browser does not treat as secure. ` +
      "Open it as http://localhost:8000 (on AI1, or through an SSH tunnel) or over https:// (see DEPLOY_GPU.md, HTTPS). Text-only tests still work.";
    warning.hidden = false;
  }

  $("sys-refresh").addEventListener("click", refresh);
  setInterval(() => {
    if (document.visibilityState === "visible" && $("system").open && $("sys-auto").checked) refresh();
  }, REFRESH_MS);
  $("system").addEventListener("toggle", () => { if ($("system").open) refresh(); });
  window.stellaStatus = { refresh };
  checkSecureContext();
  refresh();
})();
