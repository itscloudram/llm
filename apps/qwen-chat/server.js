// Local chat UI for an OpenAI-compatible endpoint behind Bifrost.
// Zero dependencies: Node 18+ (built-in fetch). Serves ./public and forwards chat calls,
// so the browser never needs CORS access to the endpoint and never sees the key.
import http from "node:http";
import { readFile, writeFile, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Readable } from "node:stream";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(ROOT, "public");
const CONFIG_FILE = process.env.CHAT_CONFIG || path.join(ROOT, "config.json");
const PORT = Number(process.env.PORT || 5178);
const HOST = "127.0.0.1"; // local only - this server holds your key

const DEFAULTS = {
  baseUrl: "", // set in Settings, e.g. https://llm.example.com/v1
  apiKey: "",
  model: "vllm/qwen3.8-27b",
  keyHeader: "x-bf-vk", // Bifrost virtual key header; "authorization" sends Bearer instead
};

async function loadConfig() {
  if (!existsSync(CONFIG_FILE)) return { ...DEFAULTS };
  try {
    return { ...DEFAULTS, ...JSON.parse(await readFile(CONFIG_FILE, "utf8")) };
  } catch (e) {
    console.error(`config.json is not valid JSON (${e.message}) - using defaults`);
    return { ...DEFAULTS };
  }
}

// Accept ".../v1", ".../v1/", ".../v1/chat/completions" or just the host.
function normalizeBase(u) {
  let s = String(u || "").trim().replace(/\/+$/, "");
  s = s.replace(/\/chat\/completions$/, "");
  if (s && !/\/v\d+$/.test(s)) s += "/v1";
  return s;
}

function authHeaders(cfg) {
  if (!cfg.apiKey) return {};
  return cfg.keyHeader === "authorization"
    ? { Authorization: `Bearer ${cfg.apiKey}` }
    : { "x-bf-vk": cfg.apiKey };
}

const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon",
};

function send(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { "Content-Type": "application/json", "Cache-Control": "no-store" });
  res.end(body);
}

async function readBody(req, limit = 20 * 1024 * 1024) { // Bifrost max_request_body_size_mb is 20
  const chunks = [];
  let size = 0;
  for await (const c of req) {
    size += c.length;
    if (size > limit) throw Object.assign(new Error("request too large (20 MB max, the gateway limit)"), { code: 413 });
    chunks.push(c);
  }
  return chunks.length ? JSON.parse(Buffer.concat(chunks).toString("utf8")) : {};
}

function friendlyNetError(e, base) {
  const c = e?.cause?.code || e?.code || "";
  if (/SELF_SIGNED|UNABLE_TO_GET_ISSUER|CERT_/.test(c))
    return `TLS certificate not trusted (${c}). On a corporate network start with NODE_EXTRA_CA_CERTS=<your-company-ca.pem> npm start`;
  if (c === "ENOTFOUND") return `Cannot resolve ${base} - check the URL and DNS`;
  if (c === "ECONNREFUSED" || c === "ECONNRESET") return `Connection to ${base} failed (${c})`;
  if (e?.name === "TimeoutError") return "The endpoint did not answer in time";
  return `${e?.message || e}${c ? ` (${c})` : ""}`;
}

async function handleChat(req, res) {
  const cfg = await loadConfig();
  const base = normalizeBase(cfg.baseUrl);
  if (!base || !cfg.apiKey) return send(res, 400, { error: { message: "Set the endpoint URL and key in Settings first." } });

  let payload;
  try {
    payload = await readBody(req);
  } catch (e) {
    return send(res, e.code === 413 ? 413 : 400, { error: { message: e.message } });
  }
  const body = { ...payload, model: payload.model || cfg.model };
  if (body.stream) body.stream_options = { include_usage: true, ...(body.stream_options || {}) };

  const ac = new AbortController();
  res.on("close", () => { if (!res.writableEnded) ac.abort(); }); // browser pressed Stop / closed tab

  const started = Date.now();
  let upstream;
  try {
    upstream = await fetch(`${base}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: body.stream ? "text/event-stream" : "application/json", ...authHeaders(cfg) },
      body: JSON.stringify(body),
      signal: ac.signal,
    });
  } catch (e) {
    if (ac.signal.aborted) return;
    return send(res, 502, { error: { message: friendlyNetError(e, base) } });
  }

  const headers = {
    "Content-Type": upstream.headers.get("content-type") || "application/json",
    "Cache-Control": "no-store",
    "X-Upstream-Status": String(upstream.status),
    "X-Proxy-Connect-Ms": String(Date.now() - started),
  };
  res.writeHead(upstream.status, headers);
  if (!upstream.body) return res.end();
  try {
    for await (const chunk of Readable.fromWeb(upstream.body)) res.write(chunk);
    res.end();
  } catch (e) {
    if (!ac.signal.aborted) res.end();
  }
}

async function handleModels(res) {
  const cfg = await loadConfig();
  const base = normalizeBase(cfg.baseUrl);
  if (!base || !cfg.apiKey) return send(res, 200, { data: [], configured: false });
  try {
    const r = await fetch(`${base}/models`, { headers: authHeaders(cfg), signal: AbortSignal.timeout(15000) });
    const j = await r.json().catch(() => ({}));
    return send(res, r.ok ? 200 : r.status, r.ok ? { data: j.data || [], configured: true } : { error: j.error || { message: `HTTP ${r.status}` } });
  } catch (e) {
    return send(res, 502, { error: { message: friendlyNetError(e, base) } });
  }
}

function publicConfig(cfg) {
  const k = cfg.apiKey || "";
  return {
    baseUrl: cfg.baseUrl, model: cfg.model, keyHeader: cfg.keyHeader,
    hasKey: Boolean(k), keyHint: k ? `${k.slice(0, 6)}…${k.slice(-4)}` : "",
    configFile: CONFIG_FILE,
  };
}

async function handleConfig(req, res) {
  if (req.method === "GET") return send(res, 200, publicConfig(await loadConfig()));
  let incoming;
  try {
    incoming = await readBody(req, 64 * 1024);
  } catch (e) {
    return send(res, 400, { error: { message: e.message } });
  }
  const cfg = await loadConfig();
  if (typeof incoming.baseUrl === "string") cfg.baseUrl = incoming.baseUrl.trim();
  if (typeof incoming.model === "string" && incoming.model.trim()) cfg.model = incoming.model.trim();
  if (incoming.keyHeader === "authorization" || incoming.keyHeader === "x-bf-vk") cfg.keyHeader = incoming.keyHeader;
  if (typeof incoming.apiKey === "string" && incoming.apiKey.trim()) cfg.apiKey = incoming.apiKey.trim(); // blank = keep current
  await writeFile(CONFIG_FILE, JSON.stringify(cfg, null, 2) + "\n", { mode: 0o600 });
  return send(res, 200, publicConfig(cfg));
}

async function serveStatic(req, res) {
  const url = new URL(req.url, "http://x");
  let p = decodeURIComponent(url.pathname);
  if (p === "/") p = "/index.html";
  const file = path.normalize(path.join(PUBLIC, p));
  if (!file.startsWith(PUBLIC)) return send(res, 403, { error: "forbidden" });
  try {
    if (!(await stat(file)).isFile()) throw new Error();
    res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream", "Cache-Control": "no-cache" });
    res.end(await readFile(file));
  } catch {
    send(res, 404, { error: "not found" });
  }
}

const server = http.createServer(async (req, res) => {
  // Only this machine's browser may call the API routes (blocks other sites from using your key).
  const origin = req.headers.origin;
  if (req.url.startsWith("/api/") && origin && !/^http:\/\/(127\.0\.0\.1|localhost):\d+$/.test(origin))
    return send(res, 403, { error: { message: "cross-origin request refused" } });
  try {
    if (req.url === "/api/chat" && req.method === "POST") return await handleChat(req, res);
    if (req.url === "/api/models" && req.method === "GET") return await handleModels(res);
    if (req.url === "/api/config" && (req.method === "GET" || req.method === "POST")) return await handleConfig(req, res);
    if (req.method === "GET") return await serveStatic(req, res);
    send(res, 405, { error: "method not allowed" });
  } catch (e) {
    console.error(e);
    if (!res.headersSent) send(res, 500, { error: { message: String(e.message || e) } });
    else res.end();
  }
});

server.listen(PORT, HOST, async () => {
  const cfg = await loadConfig();
  console.log(`\n  Qwen Chat running at  http://localhost:${PORT}`);
  console.log(`  endpoint: ${normalizeBase(cfg.baseUrl) || "(not set)"}   key: ${cfg.apiKey ? "set" : "NOT SET - open Settings"}`);
  console.log(`  config:   ${CONFIG_FILE}\n  Ctrl+C to stop\n`);
});
