// RLX dashboard front-end. Plain JS, no build step, no dependencies.
//
// State flows one way: server messages update `state`, then `render()` redraws
// both canvases from `state`. Nothing else draws.

"use strict";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const ARROWS = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // up, right, down, left as (dx, dy)
const KEYS = { ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3, w: 0, d: 1, s: 2, a: 3 };
const TILES = {
  ".": { label: "free", color: "--free" },
  "S": { label: "start", color: "--start" },
  "G": { label: "goal", color: "--goal" },
  "#": { label: "wall", color: "--wall" },
  "H": { label: "hole", color: "--hole" },
  "C": { label: "cliff", color: "--cliff" },
};
const MA_WINDOW = 50;

const state = {
  catalog: null,
  layout: [],
  V: null, Q: null, policy: null,
  trajectory: null,   // last training episode, drawn as a trail
  agentState: null,   // agent position while animating or playing
  returns: [],        // learning curve (episodes or planning iterations)
  kind: "online",
  view: "V",
  mode: "idle",       // idle | training | watching | playing | editing
  brush: "#",
  customLayout: null,
};

// ------------------------------------------------------------- socket
let ws;
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => { $("conn").textContent = "connected"; $("conn").classList.add("ok"); };
  ws.onclose = () => {
    $("conn").textContent = "disconnected, retrying…"; $("conn").classList.remove("ok");
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => onMessage(JSON.parse(ev.data));
}
const send = (msg) => ws && ws.readyState === 1 && ws.send(JSON.stringify(msg));

function onMessage(m) {
  switch (m.type) {
    case "started":
      Object.assign(state, { layout: m.layout, V: m.V, Q: m.Q, policy: m.policy,
        kind: m.kind, returns: [], trajectory: null, agentState: null, mode: "training" });
      $("chart-title").textContent = m.kind === "planning"
        ? "Greedy-policy return per sweep (mean of 10 rollouts)" : "Return per episode";
      setStatus(`${m.algo}: training…`);
      break;
    case "episodes":
      state.returns.push(...m.returns);
      Object.assign(state, { V: m.V, Q: m.Q, policy: m.policy });
      if (m.trajectory) state.trajectory = m.trajectory;
      setStatus(`episode ${state.returns.length} · last return ${fmt(m.returns.at(-1))}`);
      break;
    case "iteration":
      state.returns.push(m.eval_return ?? 0);
      Object.assign(state, { V: m.V, Q: m.Q, policy: m.policy });
      setStatus(`sweep ${m.iteration + 1} · max |ΔV| = ${m.delta.toExponential(2)}`);
      break;
    case "done":
      state.mode = "idle";
      setStatus(`${m.stopped ? "stopped" : "done"} after ${m.iterations || m.episodes} `
        + `${m.iterations ? "sweeps" : "episodes"}. Try "Watch greedy".`);
      break;
    case "rollout":
      animate(m.states, () => setStatus(`greedy episode: return ${fmt(m.ret)}, ${m.states.length - 1} steps`));
      break;
    case "play":
      if (m.layout) state.layout = m.layout;
      state.agentState = m.state;
      setStatus(m.done ? `episode over, return ${fmt(m.ret)}. Press Play to restart.`
        : `playing: arrows/WASD · reward ${fmt(m.reward)} · return ${fmt(m.ret)}`);
      if (m.done) state.mode = "idle";
      break;
    case "error":
      state.mode = state.mode === "training" ? "idle" : state.mode;
      setStatus(`⚠ ${m.message}`);
      break;
  }
  render();
}

const fmt = (x) => (x == null ? "–" : Number(x).toFixed(2).replace(/\.00$/, ""));
const setStatus = (t) => { $("status").textContent = t; };

// --------------------------------------------------------- side panel
async function init() {
  state.catalog = await (await fetch("/api/catalog")).json();

  for (const e of state.catalog.envs) $("env").add(new Option(e.label, e.id));
  $("env").add(new Option("Custom (draw your own)", "custom"));
  for (const a of state.catalog.algorithms) {
    const opt = new Option(`${a.name}${a.implemented ? "" : "  · to implement"}`, a.id);
    $("algo").add(opt);
  }
  $("algo").value = "value_iteration";

  for (const [ch, t] of Object.entries(TILES)) {
    const b = document.createElement("button");
    b.innerHTML = `<span class="swatch" style="background:var(${t.color})"></span>${t.label}`;
    b.dataset.tile = ch;
    b.onclick = () => { state.brush = ch; syncPalette(); };
    $("palette").append(b);
  }
  syncPalette();

  $("env").onchange = onEnvChange;
  $("algo").onchange = onAlgoChange;
  $("ed-rows").onchange = $("ed-cols").onchange = resizeCustom;
  $("speed").oninput = () => {
    $("speed-val").textContent = `${$("speed").value} ms`;
    send({ type: "speed", delay_ms: +$("speed").value });
  };
  $("train").onclick = startTraining;
  $("stop").onclick = () => send({ type: "stop" });
  $("watch").onclick = () => send({ type: "rollout" });
  $("play").onclick = () => { state.mode = "playing"; state.trajectory = null; send({ type: "play_reset", env: envConfig() }); };
  for (const b of $("view").querySelectorAll("button")) {
    b.onclick = () => {
      state.view = b.dataset.view;
      $("view").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      render();
    };
  }
  $("show-policy").onchange = $("show-traj").onchange = render;
  document.addEventListener("keydown", onKey);
  setupPainting();
  window.addEventListener("resize", render);

  onEnvChange();
  onAlgoChange();
  connect();
}

function syncPalette() {
  for (const b of $("palette").children) b.classList.toggle("on", b.dataset.tile === state.brush);
}

function onEnvChange() {
  const id = $("env").value;
  const custom = id === "custom";
  $("editor").hidden = !custom;
  clearAgentView();
  if (custom) {
    if (!state.customLayout) resizeCustom();
    state.layout = state.customLayout;
    state.mode = "editing";
    $("env-desc").textContent = "Draw a grid. It needs at least one S. G and H end the episode, C sends you back to S.";
  } else {
    const env = state.catalog.envs.find((e) => e.id === id);
    state.layout = env.layout;
    state.mode = "idle";
    $("env-desc").textContent = env.description;
  }
  render();
}

function onAlgoChange() {
  const a = state.catalog.algorithms.find((x) => x.id === $("algo").value);
  $("algo-desc").textContent = `${a.reference} · ${a.kind === "planning" ? "uses the model P (planning)" : "learns from experience"}`
    + (a.implemented ? "" : " · stub: write it in rlx/algorithms, then come back!");
  $("params").innerHTML = "";
  for (const [k, v] of Object.entries(a.params)) {
    const l = document.createElement("label");
    l.innerHTML = `${k} <input type="number" step="any" data-param="${k}" value="${v}">`;
    $("params").append(l);
  }
  $("episodes").disabled = a.kind === "planning";
}

function clearAgentView() {
  Object.assign(state, { V: null, Q: null, policy: null, trajectory: null, agentState: null, returns: [] });
}

function envConfig() {
  if ($("env").value !== "custom") return { id: $("env").value };
  const num = (id) => +$(id).value;
  return { custom: { layout: state.customLayout, step: num("ed-step"), goal: num("ed-goal"),
    hole: num("ed-hole"), cliff: num("ed-cliff"), slip: num("ed-slip"), max_steps: num("ed-max") } };
}

function startTraining() {
  const params = {};
  for (const input of $("params").querySelectorAll("input")) params[input.dataset.param] = +input.value;
  clearAgentView();
  send({ type: "start", env: envConfig(), algo: $("algo").value, params,
    episodes: +$("episodes").value, seed: +$("seed").value, delay_ms: +$("speed").value });
}

function onKey(e) {
  if (state.mode !== "playing" || !(e.key in KEYS)) return;
  e.preventDefault();
  send({ type: "play_step", action: KEYS[e.key] });
}

// --------------------------------------------------------- the editor
function resizeCustom() {
  const rows = Math.max(2, Math.min(16, +$("ed-rows").value));
  const cols = Math.max(2, Math.min(16, +$("ed-cols").value));
  const old = state.customLayout || [];
  state.customLayout = Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => (old[r] && old[r][c]) || ".").join(""));
  if (!old.length) { // a sensible starting map
    setCell(0, 0, "S"); setCell(rows - 1, cols - 1, "G");
  }
  if ($("env").value === "custom") { state.layout = state.customLayout; clearAgentView(); render(); }
}

function setCell(r, c, ch) {
  const row = state.customLayout[r];
  state.customLayout[r] = row.slice(0, c) + ch + row.slice(c + 1);
}

function setupPainting() {
  const canvas = $("grid");
  let down = false;
  const paint = (ev) => {
    if ($("env").value !== "custom" || state.mode === "training") return;
    const g = gridGeometry();
    const rect = canvas.getBoundingClientRect();
    const c = Math.floor((ev.clientX - rect.left - g.x0) / g.cell);
    const r = Math.floor((ev.clientY - rect.top - g.y0) / g.cell);
    if (r < 0 || c < 0 || r >= g.rows || c >= g.cols) return;
    if (state.customLayout[r][c] === state.brush) return;
    setCell(r, c, state.brush);
    state.layout = state.customLayout;
    state.mode = "editing";
    clearAgentView();
    render();
  };
  canvas.addEventListener("pointerdown", (e) => { down = true; paint(e); });
  canvas.addEventListener("pointermove", (e) => down && paint(e));
  window.addEventListener("pointerup", () => { down = false; });
}

// ------------------------------------------------------ animation
function animate(states, onEnd) {
  state.mode = "watching";
  state.trajectory = null;
  let i = 0;
  const tick = () => {
    state.agentState = states[i];
    render();
    if (++i < states.length && state.mode === "watching") setTimeout(tick, 120);
    else { state.mode = "idle"; onEnd(); }
  };
  tick();
}

// ---------------------------------------------------------- drawing
function setupCanvas(canvas, cssHeight) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  canvas.width = w * dpr;
  canvas.height = cssHeight * dpr;
  canvas.style.height = `${cssHeight}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h: cssHeight };
}

function gridGeometry() {
  const rows = state.layout.length, cols = state.layout[0].length;
  const w = $("grid").clientWidth;
  const cell = Math.floor(Math.min(w / cols, 72));
  return { rows, cols, cell, x0: Math.floor((w - cell * cols) / 2), y0: 0, h: cell * rows };
}

function valueColor(v, scale) {
  const t = Math.max(-1, Math.min(1, v / scale));
  const rgb = t >= 0 ? css("--pos") : css("--neg");
  return `rgba(${rgb}, ${0.08 + 0.8 * Math.abs(t)})`;
}

function render() {
  if (!state.layout.length) return;
  drawGrid();
  drawChart();
}

function drawGrid() {
  const g = gridGeometry();
  const { ctx } = setupCanvas($("grid"), g.h);
  const { rows, cols, cell, x0, y0 } = g;
  const cx = (c) => x0 + c * cell, cy = (r) => y0 + r * cell;
  const terminalOrWall = (ch) => ch === "#" || ch === "G" || ch === "H" || ch === "C";
  const scale = state.V ? Math.max(1e-9, ...state.V.map(Math.abs)) : 1;
  const qScale = state.Q ? Math.max(1e-9, ...state.Q.flat().map(Math.abs)) : 1;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const ch = state.layout[r][c], s = r * cols + c, x = cx(c), y = cy(r);
      ctx.fillStyle = css(TILES[ch].color);
      ctx.fillRect(x, y, cell, cell);

      if (!terminalOrWall(ch)) {
        if (state.view === "V" && state.V) {
          ctx.fillStyle = valueColor(state.V[s], scale);
          ctx.fillRect(x, y, cell, cell);
          if (cell >= 40) {
            ctx.fillStyle = css("--text");
            ctx.font = `${Math.max(9, cell / 6)}px system-ui`;
            ctx.textAlign = "center";
            ctx.fillText(fmt(state.V[s]), x + cell / 2, y + cell - 5);
          }
        } else if (state.view === "Q" && state.Q) {
          drawQTriangles(ctx, x, y, cell, state.Q[s], qScale);
        }
        if ($("show-policy").checked && state.policy) {
          const valueShown = state.view === "V" && state.V && cell >= 40;
          drawArrows(ctx, x, y - (valueShown ? cell * 0.1 : 0), cell, state.policy[s]);
        }
      }
      if (ch === "S" || ch === "G") {
        ctx.fillStyle = css("--text");
        ctx.font = `bold ${cell / 4}px system-ui`;
        ctx.textAlign = "left";
        ctx.fillText(ch, x + 4, y + cell / 4 + 2);
      }
    }
  }

  ctx.strokeStyle = css("--border");
  ctx.lineWidth = 1;
  for (let r = 0; r <= rows; r++) { ctx.beginPath(); ctx.moveTo(x0, cy(r) + 0.5); ctx.lineTo(x0 + cols * cell, cy(r) + 0.5); ctx.stroke(); }
  for (let c = 0; c <= cols; c++) { ctx.beginPath(); ctx.moveTo(cx(c) + 0.5, y0); ctx.lineTo(cx(c) + 0.5, y0 + rows * cell); ctx.stroke(); }

  const center = (s) => [cx(s % cols) + cell / 2, cy(Math.floor(s / cols)) + cell / 2];
  if ($("show-traj").checked && state.trajectory && state.trajectory.length > 1) {
    ctx.strokeStyle = css("--agent");
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = Math.max(2, cell / 14);
    ctx.lineJoin = "round";
    ctx.beginPath();
    state.trajectory.forEach((s, i) => { const [px, py] = center(s); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  if (state.agentState != null) {
    const [px, py] = center(state.agentState);
    ctx.fillStyle = css("--agent");
    ctx.beginPath(); ctx.arc(px, py, cell / 4, 0, 2 * Math.PI); ctx.fill();
  }

  $("legend").textContent = state.view === "V" && state.V
    ? `color = V(s), blue positive, red negative · |V| max ${fmt(scale)}`
    : state.view === "Q" && state.Q ? "each triangle = Q(s,a) for the action pointing that way" : "";
}

function drawQTriangles(ctx, x, y, cell, q, scale) {
  const m = [x + cell / 2, y + cell / 2];
  const corners = [[x, y], [x + cell, y], [x + cell, y + cell], [x, y + cell]];
  // action a's triangle sits on the cell edge it points to: up=top, right=right, ...
  const edges = [[0, 1], [1, 2], [2, 3], [3, 0]];
  edges.forEach(([i, j], a) => {
    ctx.fillStyle = valueColor(q[a], scale);
    ctx.beginPath(); ctx.moveTo(...m); ctx.lineTo(...corners[i]); ctx.lineTo(...corners[j]); ctx.closePath(); ctx.fill();
  });
  ctx.strokeStyle = css("--border");
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + cell, y + cell); ctx.moveTo(x + cell, y); ctx.lineTo(x, y + cell); ctx.stroke();
}

function drawArrows(ctx, x, y, cell, probs) {
  const best = Math.max(...probs);
  if (probs.every((p) => Math.abs(p - best) < 1e-6)) return; // uniform: no preference to show
  ctx.strokeStyle = ctx.fillStyle = css("--arrow");
  ctx.lineWidth = Math.max(1.5, cell / 22);
  const mx = x + cell / 2, my = y + cell / 2, len = cell * 0.24;
  probs.forEach((p, a) => {
    if (best - p > 1e-6) return; // draw every greedy action (ties included)
    const [dx, dy] = ARROWS[a];
    const ex = mx + dx * len, ey = my + dy * len;
    ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(ex, ey); ctx.stroke();
    const hs = cell * 0.09;
    ctx.beginPath();
    ctx.moveTo(ex + dx * hs, ey + dy * hs);
    ctx.lineTo(ex - dy * hs, ey + dx * hs);
    ctx.lineTo(ex + dy * hs, ey - dx * hs);
    ctx.closePath(); ctx.fill();
  });
}

function drawChart() {
  const { ctx, w, h } = setupCanvas($("chart"), 220);
  const data = state.returns;
  const pad = { l: 52, r: 12, t: 10, b: 26 };
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px system-ui";
  ctx.fillStyle = css("--muted");
  if (!data.length) { ctx.textAlign = "center"; ctx.fillText("Press Train to see the learning curve", w / 2, h / 2); return; }

  let lo = Math.min(...data), hi = Math.max(...data);
  if (lo === hi) { lo -= 1; hi += 1; }
  const n = Math.max(2, data.length);
  const X = (i) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);

  ctx.strokeStyle = css("--border");
  ctx.textAlign = "right";
  for (let k = 0; k <= 4; k++) {
    const v = lo + ((hi - lo) * k) / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(w - pad.r, Y(v)); ctx.stroke();
    ctx.fillText(fmt(v), pad.l - 6, Y(v) + 3);
  }
  ctx.textAlign = "center";
  ctx.fillText(`${data.length} ${state.kind === "planning" ? "sweeps" : "episodes"}`, (w + pad.l) / 2, h - 6);

  const line = (values, color, width) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
    values.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
    ctx.stroke();
  };
  line(data, css("--curve-raw"), 1);
  if (state.kind !== "planning") {
    let sum = 0;
    const ma = data.map((v, i) => { sum += v; if (i >= MA_WINDOW) sum -= data[i - MA_WINDOW]; return sum / Math.min(i + 1, MA_WINDOW); });
    line(ma, css("--curve"), 2);
  } else {
    line(data, css("--curve"), 2);
  }
}

init();
