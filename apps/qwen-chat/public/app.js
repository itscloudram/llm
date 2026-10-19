import { renderMarkdown } from "/markdown.js";

// ---------------------------------------------------------------- state
const LS_CHATS = "qwenchat.chats.v1";
const LS_PREFS = "qwenchat.prefs.v1";
const LS_THEME = "qwenchat.theme";
const PDFJS = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
const PDFJS_WORKER = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
const TEXT_EXT = /\.(txt|md|markdown|csv|tsv|json|jsonl|ya?ml|xml|html?|css|js|mjs|cjs|ts|tsx|jsx|py|java|kt|go|rs|rb|php|cs|c|h|cpp|hpp|sh|bash|ps1|sql|ini|toml|cfg|conf|env|log|tf|hcl|dockerfile|gradle|properties|r|scala|swift|vue|svelte)$/i;
const MAX_TEXT_BYTES = 2 * 1024 * 1024;
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;

const defaults = { contextTokens: 65536, systemPrompt: "", maxTokens: 16384, temperature: 1.0, topP: 0.95, effort: "low", stream: true, showThinking: true, vision: false, model: "" };
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch { return false; } },
};
let prefs = { ...defaults, ...store.get(LS_PREFS, {}) };
let chats = store.get(LS_CHATS, []);
let current = null;          // active chat object
let pending = [];            // attachments for the next message
let busy = null;             // { controller } while a request runs
let serverCfg = null;

const $ = (id) => document.getElementById(id);
const el = {
  app: $("app"), thread: $("thread"), empty: $("empty"), input: $("input"), composer: $("composer"), send: $("sendBtn"),
  attach: $("attachBtn"), file: $("fileInput"), atts: $("attachments"), effort: $("effortSelect"), stream: $("streamToggle"),
  tokenEst: $("tokenEst"), model: $("modelSelect"), conn: $("connDot"), list: $("chatList"), settings: $("settings"), toast: $("toast"),
};

// ---------------------------------------------------------------- helpers
const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
const fmtS = (ms) => (ms == null ? "–" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`);
const fmtN = (n) => (n == null ? "–" : n.toLocaleString());
const estTokens = (s) => Math.ceil((s || "").length / 3.6);
const kb = (b) => (b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(0)} KB` : `${(b / 1048576).toFixed(1)} MB`);
const escHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const icon = {
  copy: '<svg viewBox="0 0 20 20" aria-hidden="true"><rect x="7" y="7" width="9.5" height="9.5" rx="2"/><path d="M13 7V5.5A2 2 0 0 0 11 3.5H5.5a2 2 0 0 0-2 2V11a2 2 0 0 0 2 2H7"/></svg>',
  retry: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10a6 6 0 1 0 1.8-4.3M4 4v3.5h3.5"/></svg>',
  edit: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M12.5 4.5l3 3L8 15H5v-3z"/></svg>',
  x: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5L5 15"/></svg>',
  trash: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4.5 6h11M8 6V4.5h4V6M6 6l.7 9.5h6.6L14 6"/></svg>',
  chev: '<svg class="chev" viewBox="0 0 20 20" aria-hidden="true"><path d="M7.5 5l5 5-5 5"/></svg>',
};

function toast(msg, ms = 2600) {
  el.toast.textContent = msg;
  el.toast.classList.add("show");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => el.toast.classList.remove("show"), ms);
}

function saveChats() {
  if (store.set(LS_CHATS, chats)) return;
  // storage full: drop image data from older chats, keep the text
  for (const c of chats) for (const m of c.messages) for (const f of m.files || []) if (c !== current) delete f.dataUrl;
  if (!store.set(LS_CHATS, chats)) toast("Browser storage is full - delete some chats in Settings.");
}
const savePrefs = () => store.set(LS_PREFS, prefs);

// ---------------------------------------------------------------- theme
function applyTheme(t) {
  if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  const dark = t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  $("themeLabel").textContent = dark ? "Light theme" : "Dark theme";
}
applyTheme(store.get(LS_THEME, null));
$("themeToggle").onclick = () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  const next = dark ? "light" : "dark";
  store.set(LS_THEME, next);
  applyTheme(next);
};

// ---------------------------------------------------------------- server config + models
async function loadServerConfig() {
  try {
    serverCfg = await (await fetch("/api/config")).json();
  } catch {
    serverCfg = null;
  }
  return serverCfg;
}

async function loadModels() {
  const names = new Set();
  if (serverCfg?.model) names.add(serverCfg.model);
  let ok = false;
  let reason = "";
  try {
    const r = await fetch("/api/models");
    const j = await r.json();
    if (r.ok && j.configured) {
      ok = true;
      for (const m of j.data || []) if (m.id) names.add(m.id.includes("/") ? m.id : `${m.owned_by || "vllm"}/${m.id}`);
    } else reason = j.error?.message || (j.configured === false ? "Endpoint or key not set" : `HTTP ${r.status}`);
  } catch (e) {
    reason = e.message;
  }
  // Bifrost lists every model the gateway knows; keep only vllm/* plus the configured default.
  const list = [...names].filter((n) => n === serverCfg?.model || n.startsWith("vllm/"));
  el.model.innerHTML = list.map((n) => `<option value="${escHtml(n)}">${escHtml(n.replace(/^vllm\//, ""))}</option>`).join("") ||
    '<option value="">Set up in Settings</option>';
  const want = prefs.model && list.includes(prefs.model) ? prefs.model : serverCfg?.model;
  if (want) el.model.value = want;
  el.conn.className = `conn ${ok ? "ok" : "bad"}`;
  el.conn.title = ok ? "Connected" : `Not connected: ${reason}`;
  return { ok, reason };
}
el.model.onchange = () => { prefs.model = el.model.value; savePrefs(); };

// ---------------------------------------------------------------- sidebar
function chatGroups() {
  const now = new Date();
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const groups = [["Today", []], ["Previous 7 days", []], ["Older", []]];
  for (const c of [...chats].sort((a, b) => b.updated - a.updated)) {
    const g = c.updated >= startToday ? 0 : c.updated >= startToday - 7 * 864e5 ? 1 : 2;
    groups[g][1].push(c);
  }
  return groups.filter(([, l]) => l.length);
}

function renderList() {
  el.list.innerHTML = "";
  for (const [name, list] of chatGroups()) {
    const g = document.createElement("div");
    g.className = "group";
    g.textContent = name;
    el.list.append(g);
    for (const c of list) {
      const row = document.createElement("div");
      row.className = `chat-item${current && c.id === current.id ? " active" : ""}`;
      row.innerHTML = `<a href="#${c.id}">${escHtml(c.title || "New chat")}</a><button class="ghost del" type="button" aria-label="Delete chat">${icon.trash}</button>`;
      row.querySelector("a").onclick = (e) => { e.preventDefault(); openChat(c.id); el.app.classList.remove("open"); };
      row.querySelector(".del").onclick = () => deleteChat(c.id);
      el.list.append(row);
    }
  }
}

function deleteChat(id) {
  chats = chats.filter((c) => c.id !== id);
  saveChats();
  if (current?.id === id) newChat(); else renderList();
}

function newChat() {
  if (busy) stop();
  current = null;
  pending = [];
  renderAttachments();
  renderThread();
  renderList();
  el.input.focus();
}

function openChat(id) {
  if (busy) stop();
  current = chats.find((c) => c.id === id) || null;
  renderThread();
  renderList();
}

$("newChat").onclick = newChat;
$("toggleSidebar").onclick = () => {
  if (matchMedia("(max-width: 820px)").matches) el.app.classList.toggle("open");
  else el.app.classList.toggle("collapsed");
};
document.addEventListener("click", (e) => {
  if (el.app.classList.contains("open") && !e.target.closest(".sidebar") && !e.target.closest("#toggleSidebar")) el.app.classList.remove("open");
});

// ---------------------------------------------------------------- thread rendering
function renderThread() {
  el.thread.querySelectorAll(".msg").forEach((n) => n.remove());
  const has = current && current.messages.length;
  el.empty.hidden = Boolean(has);
  if (!has) return;
  current.messages.forEach((m, idx) => el.thread.append(m.role === "user" ? userNode(m, idx) : assistantNode(m, idx)));
  el.thread.scrollTop = el.thread.scrollHeight;
}

function fileChip(f, removable) {
  const ext = (f.name.split(".").pop() || "").slice(0, 4).toUpperCase();
  const thumb = f.kind === "image" && f.dataUrl ? `<img src="${f.dataUrl}" alt="">` : escHtml(ext || "FILE");
  const sub = f.kind === "image" ? `Image, ${kb(f.size)}` : f.kind === "pdf" ? `PDF, ${f.pages || "?"} pages, ~${fmtN(estTokens(f.text))} tokens` : `~${fmtN(estTokens(f.text))} tokens`;
  const chip = document.createElement("span");
  chip.className = `chip${f.loading ? " loading" : ""}`;
  chip.innerHTML = `<span class="ft">${thumb}</span><span class="meta"><b title="${escHtml(f.name)}">${escHtml(f.name)}</b><small>${f.loading ? "Reading…" : sub}</small></span>`;
  if (removable) {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-label", `Remove ${f.name}`);
    b.innerHTML = icon.x;
    b.onclick = () => { pending = pending.filter((p) => p !== f); renderAttachments(); };
    chip.append(b);
  }
  return chip;
}

function userNode(m, idx) {
  const n = document.createElement("div");
  n.className = "msg user";
  if (m.files?.length) {
    const fl = document.createElement("div");
    fl.className = "files";
    m.files.forEach((f) => fl.append(fileChip(f, false)));
    n.append(fl);
  }
  const b = document.createElement("div");
  b.className = "bubble";
  b.textContent = m.content || (m.files?.length ? "(attached files)" : "");
  n.append(b);
  const actions = document.createElement("div");
  actions.className = "msg-actions";
  actions.append(actionBtn(icon.copy, "Copy", () => copyText(m.content)));
  actions.append(actionBtn(icon.edit, "Edit", () => editFrom(idx)));
  n.append(actions);
  return n;
}

function actionBtn(svg, label, fn) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "ghost";
  b.innerHTML = `${svg}<span>${label}</span>`;
  b.onclick = fn;
  return b;
}

function assistantNode(m, idx) {
  const n = document.createElement("div");
  n.className = "msg assistant";
  n.dataset.idx = idx;
  fillAssistant(n, m, false);
  return n;
}

// Re-render one assistant message (used for every streamed frame and at the end).
function fillAssistant(n, m, live) {
  const thinkingOpen = n.dataset.userToggled ? n.querySelector(".thinking")?.open : undefined;
  const thinkScroll = n.querySelector(".think-text")?.scrollTop;
  let html = "";
  if (m.reasoning) {
    const live2 = live && !m.content;
    const secs = m.metrics?.thinkMs != null && m.metrics.stream ? ` for ${fmtS(m.metrics.thinkMs)}` : "";
    const label = live2 ? "Thinking" : `Thought${secs}`;
    const open = thinkingOpen ?? (live2 && prefs.showThinking);
    html += `<details class="thinking"${open ? " open" : ""}><summary>${icon.chev}${live2 ? '<span class="live-dot"></span>' : ""}<span>${label}</span></summary><div class="think-text">${escHtml(m.reasoning)}</div></details>`;
  } else if (live && !m.content) {
    html += `<div class="note">${m.metrics?.firstByteMs != null ? "Generating…" : "Waiting for the model…"}</div>`;
  }
  html += `<div class="body">${renderMarkdown(m.content || "")}</div>`;
  if (m.error) html += `<div class="err"><strong>${escHtml(m.error.title || "Request failed")}.</strong> ${escHtml(m.error.message || "")}</div>`;
  if (m.stopped) html += `<p class="note">Stopped.</p>`;
  if (!live && m.metrics) html += receipt(m.metrics);
  n.innerHTML = html;
  if (live && m.content) {
    const last = n.querySelector(".body")?.lastElementChild;
    (last && !last.classList.contains("code") && !last.classList.contains("table-wrap") ? last : n.querySelector(".body")).classList.add("cursor");
  }
  const det = n.querySelector(".thinking");
  if (det) det.querySelector("summary").onclick = () => { n.dataset.userToggled = "1"; };
  const tt = n.querySelector(".think-text");
  if (tt) tt.scrollTop = live && !m.content ? tt.scrollHeight : thinkScroll ?? 0;
  if (!live) {
    const actions = document.createElement("div");
    actions.className = "msg-actions";
    actions.append(actionBtn(icon.copy, "Copy", () => copyText(m.content)));
    actions.append(actionBtn(icon.retry, "Retry", () => retryAt(+n.dataset.idx)));
    n.append(actions);
  }
}

function receipt(t) {
  const total = t.totalMs || 0;
  const wait = t.firstByteMs ?? total;
  const thinkStart = t.firstThinkMs;
  const answerStart = t.firstAnswerMs;
  const think = thinkStart != null ? (answerStart ?? total) - thinkStart : 0;
  const answer = answerStart != null ? total - answerStart : 0;
  const pct = (v) => (total ? Math.max(0, (v / total) * 100) : 0);
  const rows = [];
  if (!t.stream) {
    rows.push(["Response", fmtS(total)]);
    if (t.serverMs != null) rows.push(["Gateway", fmtS(t.serverMs)]);
    if (t.completionTokens != null) rows.push(["Output", `${fmtN(t.completionTokens)} tokens${t.reasoningTokens ? ` (${fmtN(t.reasoningTokens)} thinking)` : ""}`]);
    if (t.tps) rows.push(["Speed", `${t.tps.toFixed(1)} tokens/s`]);
    if (t.promptTokens != null) rows.push(["Input", `${fmtN(t.promptTokens)} tokens`]);
    rows.push(["Mode", "non-streaming"]);
    if (t.finish && t.finish !== "stop") rows.push(["Ended by", t.finish === "length" ? "max tokens reached" : t.finish]);
    return `<div class="receipt" aria-label="Response timing"><div class="bar" title="Whole answer at once (no streaming split)"><span class="a" style="width:100%"></span></div>
    <dl>${rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl></div>`;
  }
  rows.push(["First token", fmtS(t.firstByteMs)]);
  if (think) rows.push([`<span class="key" style="background:var(--think-bar)"></span>Thinking`, fmtS(think)]);
  if (answer) rows.push([`<span class="key" style="background:var(--answer-bar)"></span>Answer`, fmtS(answer)]);
  rows.push(["Total", fmtS(total)]);
  if (t.completionTokens != null) rows.push(["Output", `${fmtN(t.completionTokens)} tokens${t.reasoningTokens ? ` (${fmtN(t.reasoningTokens)} thinking)` : ""}`]);
  if (t.tps) rows.push(["Speed", `${t.tps.toFixed(1)} tokens/s`]);
  if (t.promptTokens != null) rows.push(["Input", `${fmtN(t.promptTokens)} tokens`]);
  if (t.serverMs != null) rows.push(["Gateway", fmtS(t.serverMs)]);
  if (t.finish && t.finish !== "stop") rows.push(["Ended by", t.finish === "length" ? "max tokens reached" : t.finish]);
  return `<div class="receipt" aria-label="Response timing">
    <div class="bar" title="Waiting, thinking and answering, to scale"><span class="w" style="width:${pct(wait)}%"></span><span class="t" style="width:${pct(think)}%"></span><span class="a" style="width:${pct(answer)}%"></span></div>
    <dl>${rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl></div>`;
}

async function copyText(s) {
  try { await navigator.clipboard.writeText(s || ""); toast("Copied"); } catch { toast("Copy failed - select the text instead"); }
}
el.thread.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy-code]");
  if (!b) return;
  copyText(b.closest(".code").querySelector("code").textContent);
  b.textContent = "Copied";
  setTimeout(() => (b.textContent = "Copy"), 1400);
});

// ---------------------------------------------------------------- attachments
function renderAttachments() {
  el.atts.innerHTML = "";
  pending.forEach((f) => el.atts.append(fileChip(f, true)));
  updateTokenEstimate();
}

async function loadPdfJs() {
  if (window.pdfjsLib) return window.pdfjsLib;
  await new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = PDFJS;
    s.onload = res;
    s.onerror = () => rej(new Error("Could not load the PDF reader from cdnjs.cloudflare.com (no internet or blocked by proxy)."));
    document.head.append(s);
  });
  window.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER;
  return window.pdfjsLib;
}

async function readPdf(file) {
  const lib = await loadPdfJs();
  const doc = await lib.getDocument({ data: await file.arrayBuffer() }).promise;
  const pages = [];
  for (let p = 1; p <= doc.numPages; p++) {
    const tc = await (await doc.getPage(p)).getTextContent();
    pages.push(`--- page ${p} ---\n` + tc.items.map((it) => it.str + (it.hasEOL ? "\n" : " ")).join("").replace(/[ \t]+\n/g, "\n").trim());
  }
  const text = pages.join("\n\n");
  if (!text.replace(/--- page \d+ ---/g, "").trim()) throw new Error("This PDF has no selectable text (it is probably scanned). Export it as text or use OCR first.");
  return { text, pages: doc.numPages };
}

const readAsDataUrl = (file) => new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result); r.onerror = () => rej(r.error); r.readAsDataURL(file); });

async function addFiles(files) {
  for (const file of files) {
    const name = file.name || "pasted-image.png";
    const isImage = file.type.startsWith("image/");
    const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(name);
    const isText = !isImage && !isPdf && (file.type.startsWith("text/") || TEXT_EXT.test(name) || /^(dockerfile|makefile|readme)$/i.test(name) ||
      ["application/json", "application/xml", "application/x-yaml", "application/javascript", "application/sql"].includes(file.type));
    if (isImage && !prefs.vision) { toast(`${name}: turn on "Send attached images" in Settings if your model accepts images.`, 4200); continue; }
    if (!isImage && !isPdf && !isText) { toast(`${name}: this file type can't be read. Attach text, code, CSV, JSON or PDF (save Word/Excel files as PDF or CSV).`, 4800); continue; }
    if (isImage && file.size > MAX_IMAGE_BYTES) { toast(`${name} is larger than ${kb(MAX_IMAGE_BYTES)}.`); continue; }
    if (isText && file.size > MAX_TEXT_BYTES) { toast(`${name} is larger than ${kb(MAX_TEXT_BYTES)} - too big for the model's context.`); continue; }

    const entry = { id: uid(), name, size: file.size, kind: isImage ? "image" : isPdf ? "pdf" : "text", loading: true };
    pending.push(entry);
    renderAttachments();
    try {
      if (isImage) entry.dataUrl = await readAsDataUrl(file);
      else if (isPdf) Object.assign(entry, await readPdf(file));
      else entry.text = await file.text();
      entry.loading = false;
    } catch (e) {
      pending = pending.filter((p) => p !== entry);
      toast(`${name}: ${e.message}`, 5000);
    }
    renderAttachments();
  }
}

el.attach.onclick = () => el.file.click();
el.file.onchange = () => { addFiles([...el.file.files]); el.file.value = ""; };
["dragenter", "dragover"].forEach((t) => el.composer.addEventListener(t, (e) => { if ([...e.dataTransfer.types].includes("Files")) { e.preventDefault(); el.composer.classList.add("dragging"); } }));
["dragleave", "drop"].forEach((t) => el.composer.addEventListener(t, (e) => { if (t === "dragleave" && el.composer.contains(e.relatedTarget)) return; el.composer.classList.remove("dragging"); }));
el.composer.addEventListener("drop", (e) => { e.preventDefault(); addFiles([...e.dataTransfer.files]); });
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => { if (!el.composer.contains(e.target)) { e.preventDefault(); addFiles([...e.dataTransfer.files]); } });
el.input.addEventListener("paste", (e) => {
  const files = [...(e.clipboardData?.files || [])];
  if (files.length) { e.preventDefault(); addFiles(files); }
});

// ---------------------------------------------------------------- composer
function autosize() {
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(el.input.scrollHeight, innerHeight * 0.4)}px`;
}

function historyTokens() {
  let t = estTokens(prefs.systemPrompt);
  for (const m of current?.messages || []) t += estTokens(m.content) + (m.files || []).reduce((a, f) => a + estTokens(f.text), 0);
  return t;
}

function updateTokenEstimate() {
  const t = historyTokens() + estTokens(el.input.value) + pending.reduce((a, f) => a + estTokens(f.text), 0);
  const ctx = Number(prefs.contextTokens) || 65536;
  const budget = ctx - (prefs.maxTokens || 0);
  el.tokenEst.textContent = t > 200 ? `~${fmtN(t)} of ${fmtN(ctx)} tokens` : "";
  el.tokenEst.classList.toggle("warn", t > budget);
  el.tokenEst.title = t > budget ? `Prompt plus max tokens may exceed the ${fmtN(ctx)}-token context. Start a new chat or attach less.` : "Rough estimate of the context used";
}
el.input.addEventListener("input", () => { autosize(); updateTokenEstimate(); });
el.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); el.composer.requestSubmit(); }
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && busy && !el.settings.open) stop(); });

el.effort.value = prefs.effort;
el.effort.onchange = () => { prefs.effort = el.effort.value; savePrefs(); };
el.stream.checked = prefs.stream;
el.stream.onchange = () => { prefs.stream = el.stream.checked; savePrefs(); };
$("starters").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-prompt]");
  if (!b) return;
  el.input.value = b.dataset.prompt;
  autosize();
  el.input.focus();
});

el.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  if (busy) return stop();
  const text = el.input.value.trim();
  if (!text && !pending.length) return;
  if (pending.some((f) => f.loading)) return toast("Still reading an attachment…");
  if (!serverCfg?.hasKey) { toast("Add the endpoint URL and key in Settings first."); return openSettings(); }
  if (!current) {
    current = { id: uid(), title: (text || pending[0]?.name || "New chat").slice(0, 60), messages: [], created: Date.now(), updated: Date.now() };
    chats.push(current);
  }
  current.messages.push({ role: "user", content: text, files: pending.map(({ loading, ...f }) => f) });
  pending = [];
  el.input.value = "";
  autosize();
  renderAttachments();
  run();
});

function editFrom(idx) {
  if (busy) return;
  const m = current.messages[idx];
  el.input.value = m.content;
  pending = (m.files || []).map((f) => ({ ...f }));
  current.messages = current.messages.slice(0, idx);
  saveChats();
  renderThread();
  renderAttachments();
  autosize();
  el.input.focus();
}

function retryAt(idx) {
  if (busy) return;
  current.messages = current.messages.slice(0, idx); // drop this answer and anything after it
  run();
}

// ---------------------------------------------------------------- request building
function apiMessages() {
  const out = [];
  if (prefs.systemPrompt.trim()) out.push({ role: "system", content: prefs.systemPrompt.trim() });
  for (const m of current.messages) {
    if (m.role === "assistant") {
      if (m.content) out.push({ role: "assistant", content: m.content });
      continue;
    }
    const docs = (m.files || []).filter((f) => f.text);
    const imgs = (m.files || []).filter((f) => f.kind === "image" && f.dataUrl);
    let text = docs.map((f) => `<document name="${f.name}">\n${f.text}\n</document>`).join("\n\n");
    text = [text, m.content].filter(Boolean).join("\n\n");
    if (imgs.length && prefs.vision) {
      out.push({ role: "user", content: [{ type: "text", text: text || "Describe the image." }, ...imgs.map((f) => ({ type: "image_url", image_url: { url: f.dataUrl } }))] });
    } else out.push({ role: "user", content: text });
  }
  return out;
}

function payload() {
  const p = { model: el.model.value || serverCfg?.model, messages: apiMessages(), stream: prefs.stream, max_tokens: Number(prefs.maxTokens) || 4096 };
  if (prefs.temperature !== "" && prefs.temperature != null) p.temperature = Number(prefs.temperature);
  if (prefs.topP !== "" && prefs.topP != null) p.top_p = Number(prefs.topP);
  if (prefs.effort) p.reasoning_effort = prefs.effort;
  return p;
}

// ---------------------------------------------------------------- run one turn
function setBusy(on) {
  el.send.classList.toggle("busy", on);
  el.send.setAttribute("aria-label", on ? "Stop" : "Send");
}

function stop() {
  busy?.controller.abort();
}

function errorFrom(status, j, text) {
  const msg = j?.error?.message || j?.error?.error || j?.message || (typeof j?.error === "string" ? j.error : "") || text || `HTTP ${status}`;
  const titles = { 400: "The request was rejected", 401: "The key was not accepted", 403: "Access denied", 404: "Endpoint not found", 413: "Request too large", 429: "Rate limited", 500: "The model returned an error", 502: "The gateway could not reach the model", 503: "The model is not ready", 504: "The model took too long" };
  let hint = "";
  if (status === 401) hint = " Check the virtual key in Settings.";
  if (status === 500 && /reasoning|effort/i.test(msg)) hint = " Try another Thinking level.";
  if (status === 400 && /context|maximum|length|tokens/i.test(msg)) hint = " Start a new chat, attach less, or lower Max tokens.";
  if (status === 502 || status === 503) hint = " Is the EC2 instance running? It needs about 3 minutes after a start.";
  return { title: titles[status] || `HTTP ${status}`, message: msg + hint };
}

async function run() {
  current.updated = Date.now();
  const msg = { role: "assistant", content: "", reasoning: "", metrics: null };
  current.messages.push(msg);
  saveChats();
  renderThread();
  renderList();

  const node = [...el.thread.querySelectorAll(".msg.assistant")].pop();
  const t = { stream: prefs.stream };
  const t0 = performance.now();
  const since = () => performance.now() - t0;
  msg.metrics = t;
  const controller = new AbortController();
  busy = { controller };
  setBusy(true);

  let raf = 0;
  const paint = () => {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      const nearBottom = el.thread.scrollHeight - el.thread.scrollTop - el.thread.clientHeight < 140;
      fillAssistant(node, msg, true);
      if (nearBottom) el.thread.scrollTop = el.thread.scrollHeight;
    });
  };
  paint();

  const take = (j) => {
    if (j.error) throw Object.assign(new Error("stream error"), { api: errorFrom(Number(j.error.code) || Number(j.status_code) || 500, j, "") });
    const ch = j.choices?.[0];
    const d = ch?.delta || ch?.message || {};
    const r = d.reasoning ?? d.reasoning_content;
    if (r) { if (t.firstThinkMs == null) t.firstThinkMs = since(); msg.reasoning += r; }
    if (d.content) { if (t.firstAnswerMs == null) t.firstAnswerMs = since(); msg.content += d.content; }
    if (ch?.finish_reason) t.finish = ch.finish_reason;
    if (j.usage) {
      t.promptTokens = j.usage.prompt_tokens;
      t.completionTokens = j.usage.completion_tokens;
      t.reasoningTokens = j.usage.completion_tokens_details?.reasoning_tokens;
    }
    const lat = j.extra_fields?.latency;
    if (typeof lat === "number") t.serverMs = lat;
  };

  try {
    const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload()), signal: controller.signal });
    if (!res.ok) {
      const text = await res.text();
      let j = null;
      try { j = JSON.parse(text); } catch { /* not JSON */ }
      throw Object.assign(new Error("http"), { api: errorFrom(res.status, j, text.slice(0, 400)) });
    }
    const ctype = res.headers.get("content-type") || "";
    if (ctype.includes("text/event-stream")) {
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (t.firstByteMs == null) t.firstByteMs = since();
        buf += dec.decode(value, { stream: true });
        const events = buf.split(/\r?\n\r?\n/);
        buf = events.pop();
        for (const ev of events) {
          const data = ev.split(/\r?\n/).filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trimStart()).join("\n");
          if (!data || data === "[DONE]") continue;
          let j;
          try { j = JSON.parse(data); } catch { continue; }
          take(j);
        }
        paint();
      }
    } else {
      const j = await res.json();
      t.firstByteMs = since();
      take(j);
      if (t.firstThinkMs != null) t.firstThinkMs = t.firstByteMs; // non-streaming: no split available
      if (t.firstAnswerMs != null) t.firstAnswerMs = t.firstByteMs;
    }
    if (!msg.content && !msg.error) {
      msg.error = t.finish === "length"
        ? { title: "No answer yet", message: "Max tokens ran out while the model was still thinking. Raise Max tokens in Settings or choose a lower Thinking level, then Retry." }
        : { title: "Empty answer", message: "The model returned no text." };
    }
  } catch (e) {
    if (controller.signal.aborted) msg.stopped = true;
    else msg.error = e.api || { title: "Could not reach the local server", message: `${e.message}. Is "npm start" still running?` };
  } finally {
    t.totalMs = since();
    if (t.firstThinkMs != null) t.thinkMs = (t.firstAnswerMs ?? t.totalMs) - t.firstThinkMs;
    const genStart = t.firstThinkMs ?? t.firstAnswerMs;
    if (t.completionTokens && genStart != null && t.stream) t.tps = t.completionTokens / Math.max(0.001, (t.totalMs - genStart) / 1000);
    else if (t.completionTokens && !t.stream && t.serverMs) t.tps = t.completionTokens / (t.serverMs / 1000);
    busy = null;
    setBusy(false);
    cancelAnimationFrame(raf);
    const nearBottom = el.thread.scrollHeight - el.thread.scrollTop - el.thread.clientHeight < 200;
    fillAssistant(node, msg, false);
    if (nearBottom) el.thread.scrollTop = el.thread.scrollHeight;
    current.updated = Date.now();
    saveChats();
    renderList();
    updateTokenEstimate();
  }
}

// ---------------------------------------------------------------- settings
function openSettings() {
  $("cfgUrl").value = serverCfg?.baseUrl || "";
  $("cfgKey").value = "";
  $("cfgKey").placeholder = serverCfg?.hasKey ? `Saved (${serverCfg.keyHint}) - leave blank to keep` : "sk-bf-…";
  $("cfgKeyHint").textContent = serverCfg?.hasKey ? "Paste a new key only to replace the saved one." : "Create one on the instance: sudo python3 llm-admin.py create my-laptop";
  $("cfgHeader").value = serverCfg?.keyHeader || "x-bf-vk";
  $("cfgModel").value = serverCfg?.model || "";
  $("cfgPath").textContent = serverCfg?.configFile || "config.json";
  $("setSystem").value = prefs.systemPrompt;
  $("setMaxTokens").value = prefs.maxTokens;
  $("setContext").value = prefs.contextTokens;
  $("setTemp").value = prefs.temperature;
  $("setTopP").value = prefs.topP;
  $("setShowThinking").checked = prefs.showThinking;
  $("setVision").checked = prefs.vision;
  $("testResult").textContent = "";
  $("testResult").className = "";
  el.settings.showModal();
}

async function saveConnection() {
  const body = { baseUrl: $("cfgUrl").value.trim(), apiKey: $("cfgKey").value.trim(), keyHeader: $("cfgHeader").value, model: $("cfgModel").value.trim() };
  const r = await fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error?.message || `HTTP ${r.status}`);
  serverCfg = await r.json();
  $("cfgKey").value = "";
  $("cfgKey").placeholder = serverCfg.hasKey ? `Saved (${serverCfg.keyHint}) - leave blank to keep` : "sk-bf-…";
}

function savePrefsFromForm() {
  prefs.systemPrompt = $("setSystem").value;
  prefs.maxTokens = Math.max(64, Math.min(131072, Number($("setMaxTokens").value) || defaults.maxTokens));
  prefs.contextTokens = Math.max(4096, Math.min(262144, Number($("setContext").value) || defaults.contextTokens));
  prefs.temperature = $("setTemp").value === "" ? "" : Number($("setTemp").value);
  prefs.topP = $("setTopP").value === "" ? "" : Number($("setTopP").value);
  prefs.showThinking = $("setShowThinking").checked;
  prefs.vision = $("setVision").checked;
  savePrefs();
  updateTokenEstimate();
}

$("openSettings").onclick = openSettings;
$("testConn").onclick = async () => {
  const out = $("testResult");
  out.className = "";
  out.textContent = "Testing…";
  try {
    await saveConnection();
    const { ok, reason } = await loadModels();
    out.className = ok ? "ok" : "bad";
    out.textContent = ok ? `Connected. ${el.model.options.length} model(s) available.` : reason;
  } catch (e) {
    out.className = "bad";
    out.textContent = e.message;
  }
};
$("settingsForm").addEventListener("submit", async (e) => {
  if (e.submitter?.value !== "save") return;
  e.preventDefault();
  try {
    await saveConnection();
    savePrefsFromForm();
    el.settings.close();
    const { ok, reason } = await loadModels();
    toast(ok ? "Settings saved. Connected." : `Settings saved, but not connected: ${reason}`, ok ? 2200 : 5000);
  } catch (err) {
    $("testResult").className = "bad";
    $("testResult").textContent = err.message;
  }
});
$("clearChats").onclick = () => {
  if (!confirm("Delete all chats from this browser? This can't be undone.")) return;
  chats = [];
  saveChats();
  el.settings.close();
  newChat();
};

// ---------------------------------------------------------------- start
function setGreeting() {
  const h = new Date().getHours();
  $("greeting").textContent = h < 5 ? "Working late?" : h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
}

(async function init() {
  setGreeting();
  renderList();
  renderThread();
  autosize();
  await loadServerConfig();
  if (!serverCfg) {
    el.conn.className = "conn bad";
    toast('The local server is not running. Start it with "npm start".', 6000);
    return;
  }
  const { ok } = await loadModels();
  if (!serverCfg.hasKey) openSettings();
  else if (!ok) toast("Can't reach the endpoint - check Settings, or start the EC2 instance.", 5000);
  el.input.focus();
})();
