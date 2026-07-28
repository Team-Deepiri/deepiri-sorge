/**
 * Sorge webhook dispatcher — triggers review only when /sorge slash command is used
 * on a PR comment (issue_comment event).
 *
 * Also hosts the escalate ticket ledger (KV) and cron → sorge-drain dispatch.
 *
 * Env vars:
 *   GITHUB_WEBHOOK_SECRET    — HMAC secret for verifying webhook payloads
 *   SORGE_APP_ID             — GitHub App ID
 *   SORGE_APP_PRIVATE_KEY    — GitHub App private key, PKCS#8 PEM
 *   SORGE_DISPATCH_REPO      — Target repo for repository_dispatch (default: Team-Deepiri/deepiri-sorge)
 *   SORGE_BOT_LOGIN          — Extra command handle if bot login is not "sorge"
 *   SORGE_LEDGER_SECRET      — Bearer token for /ledger/* from Actions
 *
 * Bindings:
 *   SORGE_LEDGER (KV)        — escalate ticket store
 *
 * Deploy: npx wrangler deploy
 */

const DISPATCH_EVENT_REVIEW = "sorge-review";
const DISPATCH_EVENT_DRAIN = "sorge-drain";

async function verifySignature(body, signature, secret) {
  if (!secret || !signature) return false;
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["verify"],
  );
  const sigBytes = new Uint8Array(
    signature
      .replace(/^sha256=/, "")
      .match(/.{2}/g)
      .map((b) => parseInt(b, 16)),
  );
  return crypto.subtle.verify("HMAC", key, sigBytes, enc.encode(body));
}

import { hasSorgeSlashCommand } from "./slash_command.js";

export { hasSorgeSlashCommand };

function base64UrlEncode(bytes) {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary).replace(/=/g, "").replace(/\+/g, "-").replace(/\//g, "_");
}

function base64UrlEncodeString(str) {
  return base64UrlEncode(new TextEncoder().encode(str));
}

function pemToArrayBuffer(pem) {
  const b64 = pem
    .replace(/-----BEGIN [^-]+-----/, "")
    .replace(/-----END [^-]+-----/, "")
    .replace(/\s+/g, "");
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

async function createAppJwt(appId, privateKeyPem) {
  const now = Math.floor(Date.now() / 1000);
  const header = { alg: "RS256", typ: "JWT" };
  const payload = {
    iat: now - 60,
    exp: now + 600,
    iss: appId,
  };

  const encodedHeader = base64UrlEncodeString(JSON.stringify(header));
  const encodedPayload = base64UrlEncodeString(JSON.stringify(payload));
  const signingInput = `${encodedHeader}.${encodedPayload}`;

  const key = await crypto.subtle.importKey(
    "pkcs8",
    pemToArrayBuffer(privateKeyPem),
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["sign"],
  );

  const signature = await crypto.subtle.sign(
    "RSASSA-PKCS1-v1_5",
    key,
    new TextEncoder().encode(signingInput),
  );

  const encodedSignature = base64UrlEncode(new Uint8Array(signature));
  return `${signingInput}.${encodedSignature}`;
}

async function getInstallationToken(env, installationId, repoName) {
  const jwt = await createAppJwt(env.SORGE_APP_ID, env.SORGE_APP_PRIVATE_KEY);

  const res = await fetch(
    `https://api.github.com/app/installations/${installationId}/access_tokens`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${jwt}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "sorge-webhook-worker",
      },
      body: JSON.stringify({
        repositories: [repoName],
      }),
    },
  );

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`installation token request failed: ${res.status} ${text}`);
  }

  const data = await res.json();
  return data.token;
}

async function dispatchEvent(env, eventType, clientPayload = {}) {
  if (!env.SORGE_APP_ID || !env.SORGE_APP_PRIVATE_KEY) {
    return { error: "SORGE_APP_ID or SORGE_APP_PRIVATE_KEY not configured" };
  }

  const dispatchRepo = env.SORGE_DISPATCH_REPO || "Team-Deepiri/deepiri-sorge";
  const dispatchRepoName = dispatchRepo.split("/").pop();
  // Prefer org installation id from payload when present; else look up via JWT list
  let installationId = clientPayload.installation_id;
  if (!installationId) {
    // Use first installation that can access dispatch repo — caller should pass id for review
    return { error: "installation_id required for dispatch" };
  }

  let token;
  try {
    token = await getInstallationToken(env, installationId, dispatchRepoName);
  } catch (err) {
    return { error: `failed to mint installation token: ${err.message || err}` };
  }

  const res = await fetch(`https://api.github.com/repos/${dispatchRepo}/dispatches`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "Content-Type": "application/json",
      "User-Agent": "sorge-webhook-worker",
    },
    body: JSON.stringify({
      event_type: eventType,
      client_payload: clientPayload,
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    return { error: `dispatch failed: ${res.status} ${text}` };
  }

  return { ok: true, event_type: eventType };
}

async function dispatchReview(env, { repo, prNumber, installationId, trigger }) {
  if (!repo || !prNumber || !installationId) {
    return { skipped: true, reason: "missing repo, pr_number, or installation" };
  }
  return dispatchEvent(env, DISPATCH_EVENT_REVIEW, {
    repo,
    pr_number: prNumber,
    installation_id: installationId,
    trigger: trigger || "slash_command",
    force: trigger === "slash_command",
  });
}

async function dispatchDrain(env, installationId) {
  if (!installationId) {
    return { skipped: true, reason: "missing installation_id for drain dispatch" };
  }
  return dispatchEvent(env, DISPATCH_EVENT_DRAIN, {
    installation_id: installationId,
    trigger: "cron",
  });
}

function handleIssueComment(payload, env) {
  if (payload.action !== "created") {
    return { skipped: true, reason: "not a new comment", action: payload.action };
  }

  const issue = payload.issue;
  if (!issue?.pull_request) {
    return { skipped: true, reason: "comment is not on a pull request" };
  }

  const comment = payload.comment;
  if (!comment?.body) {
    return { skipped: true, reason: "empty comment" };
  }

  if (comment.user?.type === "Bot") {
    return { skipped: true, reason: "ignore bot comments" };
  }

  if (!hasSorgeSlashCommand(comment.body, env.SORGE_BOT_LOGIN)) {
    return { skipped: true, reason: "no /sorge slash command" };
  }

  return dispatchReview(env, {
    repo: payload.repository.full_name,
    prNumber: issue.number,
    installationId: payload.installation?.id,
    trigger: "slash_command",
  });
}

function assertLedgerAuth(request, env) {
  const auth = request.headers.get("Authorization") || "";
  const token = auth.replace(/^Bearer\s+/i, "");
  return Boolean(env.SORGE_LEDGER_SECRET && token === env.SORGE_LEDGER_SECRET);
}

async function loadLedger(env) {
  if (!env.SORGE_LEDGER) return { tickets: [] };
  const raw = await env.SORGE_LEDGER.get("tickets");
  if (!raw) return { tickets: [] };
  try {
    return JSON.parse(raw);
  } catch {
    return { tickets: [] };
  }
}

async function saveLedger(env, data) {
  if (!env.SORGE_LEDGER) return;
  await env.SORGE_LEDGER.put("tickets", JSON.stringify(data));
}

async function loadQuota(env) {
  if (!env.SORGE_LEDGER) return null;
  const raw = await env.SORGE_LEDGER.get("quota_daily");
  if (!raw) return { date: utcToday(), used: {} };
  try {
    const data = JSON.parse(raw);
    if (data.date !== utcToday()) return { date: utcToday(), used: {} };
    return { date: data.date, used: data.used || {} };
  } catch {
    return { date: utcToday(), used: {} };
  }
}

async function saveQuota(env, data) {
  if (!env.SORGE_LEDGER) return;
  await env.SORGE_LEDGER.put("quota_daily", JSON.stringify(data));
}

function utcToday() {
  return new Date().toISOString().slice(0, 10);
}

function mergeUsed(a, b) {
  const out = { ...(a || {}) };
  for (const [k, v] of Object.entries(b || {})) {
    const n = Number(v) || 0;
    out[k] = Math.max(Number(out[k]) || 0, n);
  }
  return out;
}

async function loadProviderStatus(env) {
  if (!env.SORGE_LEDGER) return null;
  const raw = await env.SORGE_LEDGER.get("provider_status");
  if (!raw) return { cooldowns: {} };
  try {
    const data = JSON.parse(raw);
    const now = Date.now() / 1000;
    const cool = {};
    for (const [k, v] of Object.entries(data.cooldowns || {})) {
      const until = Number(v) || 0;
      if (until > now) cool[k] = until;
    }
    return { cooldowns: cool };
  } catch {
    return { cooldowns: {} };
  }
}

async function saveProviderStatus(env, data) {
  if (!env.SORGE_LEDGER) return;
  await env.SORGE_LEDGER.put("provider_status", JSON.stringify(data));
}

function mergeCooldowns(a, b) {
  const out = { ...(a || {}) };
  for (const [k, v] of Object.entries(b || {})) {
    const n = Number(v) || 0;
    out[k] = Math.max(Number(out[k]) || 0, n);
  }
  return out;
}

async function loadSlots(env) {
  if (!env.SORGE_LEDGER) return {};
  const raw = await env.SORGE_LEDGER.get("provider_slots");
  if (!raw) return {};
  try {
    return JSON.parse(raw) || {};
  } catch {
    return {};
  }
}

async function saveSlots(env, slots) {
  if (!env.SORGE_LEDGER) return;
  await env.SORGE_LEDGER.put("provider_slots", JSON.stringify(slots));
}

async function loadRetries(env) {
  if (!env.SORGE_LEDGER) return [];
  const raw = await env.SORGE_LEDGER.get("review_retries");
  if (!raw) return [];
  try {
    const data = JSON.parse(raw);
    return Array.isArray(data) ? data : data.retries || [];
  } catch {
    return [];
  }
}

async function saveRetries(env, retries) {
  if (!env.SORGE_LEDGER) return;
  const cutoff = Date.now() / 1000 - 86400;
  const pruned = retries.filter(
    (r) => r.status === "pending" || Number(r.created_at || 0) > cutoff,
  );
  await env.SORGE_LEDGER.put("review_retries", JSON.stringify(pruned));
}

async function flushDueRetries(env) {
  const retries = await loadRetries(env);
  const now = Date.now() / 1000;
  const due = [];
  for (const r of retries) {
    if (r.status !== "pending") continue;
    if (Number(r.not_before) > now) continue;
    r.status = "dispatched";
    r.dispatched_at = now;
    due.push(r);
  }
  if (!due.length) {
    return { dispatched: 0 };
  }
  await saveRetries(env, retries);
  const results = [];
  for (const r of due) {
    const out = await dispatchEvent(env, DISPATCH_EVENT_REVIEW, {
      repo: r.repo,
      pr_number: r.pr_number,
      installation_id: r.installation_id,
      trigger: "auto_retry",
      force: true,
      auto_retry: true,
      comment_id: r.comment_id || null,
    });
    results.push({ repo: r.repo, pr: r.pr_number, ...out });
  }
  console.log(`flushed ${due.length} due auto-retr(y/ies)`, JSON.stringify(results));
  return { dispatched: due.length, results };
}

async function handleLedger(request, env, url) {
  if (!assertLedgerAuth(request, env)) {
    return new Response(JSON.stringify({ error: "unauthorized" }), { status: 401 });
  }
  if (!env.SORGE_LEDGER) {
    return new Response(JSON.stringify({ error: "KV not bound" }), { status: 503 });
  }

  const data = await loadLedger(env);

  if (url.pathname === "/ledger/tickets" && request.method === "POST") {
    const body = await request.json();
    const incoming = body.tickets || [];
    const repos = new Set(
      incoming
        .filter((t) => t.repo && t.pr_number)
        .map((t) => `${t.repo}#${t.pr_number}`),
    );
    for (const t of data.tickets) {
      if (t.status === "pending" && repos.has(`${t.repo}#${t.pr_number}`)) {
        t.status = "cancelled";
      }
    }
    for (const t of incoming) {
      data.tickets.push({ ...t, status: t.status || "pending" });
    }
    await saveLedger(env, data);
    return json({ enqueued: incoming.length });
  }

  if (url.pathname === "/ledger/tickets" && request.method === "GET") {
    if (url.searchParams.get("count_only") === "1") {
      const pending = data.tickets.filter((t) => t.status === "pending").length;
      return json({ pending });
    }
    const limit = Math.max(1, Math.min(20, parseInt(url.searchParams.get("limit") || "8", 10)));
    const claimed = [];
    const now = Date.now() / 1000;
    for (const t of data.tickets) {
      if (claimed.length >= limit) break;
      if (t.status !== "pending") continue;
      t.status = "claimed";
      t.claimed_at = now;
      claimed.push(t);
    }
    await saveLedger(env, data);
    return json({ tickets: claimed });
  }

  if (url.pathname === "/ledger/tickets/ack" && request.method === "POST") {
    const body = await request.json();
    const ids = new Set(body.ticket_ids || []);
    const status = body.status || "done";
    for (const t of data.tickets) {
      if (ids.has(t.ticket_id)) t.status = status;
    }
    await saveLedger(env, data);
    return json({ acked: ids.size });
  }

  if (url.pathname === "/ledger/tickets/cancel" && request.method === "POST") {
    const body = await request.json();
    let n = 0;
    for (const t of data.tickets) {
      if (
        t.repo === body.repo &&
        Number(t.pr_number) === Number(body.pr_number) &&
        t.status === "pending"
      ) {
        t.status = "cancelled";
        n += 1;
      }
    }
    await saveLedger(env, data);
    return json({ cancelled: n });
  }

  if (url.pathname === "/ledger/tickets/attach_comment" && request.method === "POST") {
    const body = await request.json();
    let n = 0;
    for (const t of data.tickets) {
      if (
        t.repo === body.repo &&
        Number(t.pr_number) === Number(body.pr_number) &&
        (t.status === "pending" || t.status === "claimed")
      ) {
        t.comment_id = body.comment_id;
        n += 1;
      }
    }
    await saveLedger(env, data);
    return json({ updated: n });
  }

  // Shared soft RPD across Actions runs (max-merge by UTC day).
  if (url.pathname === "/ledger/quota" && request.method === "GET") {
    const q = (await loadQuota(env)) || { date: utcToday(), used: {} };
    return json(q);
  }

  if (url.pathname === "/ledger/quota" && request.method === "POST") {
    const body = await request.json();
    const current = (await loadQuota(env)) || { date: utcToday(), used: {} };
    const merged = {
      date: utcToday(),
      used: mergeUsed(current.used, body.used || {}),
      updated_at: new Date().toISOString(),
    };
    await saveQuota(env, merged);
    return json(merged);
  }

  // Cross-run provider cooldowns (max-merge cooldown_until timestamps).
  if (url.pathname === "/ledger/provider_status" && request.method === "GET") {
    const status = (await loadProviderStatus(env)) || { cooldowns: {} };
    return json(status);
  }

  if (url.pathname === "/ledger/provider_status" && request.method === "POST") {
    const body = await request.json();
    const current = (await loadProviderStatus(env)) || { cooldowns: {} };
    const merged = {
      cooldowns: mergeCooldowns(current.cooldowns, body.cooldowns || {}),
      updated_at: new Date().toISOString(),
    };
    await saveProviderStatus(env, merged);
    return json(merged);
  }

  // Soft concurrency semaphore across Actions runs.
  if (url.pathname === "/ledger/slots/acquire" && request.method === "POST") {
    const body = await request.json();
    const provider = String(body.provider || "");
    const holderId = String(body.holder_id || "");
    const maxInflight = Math.max(1, Number(body.max_inflight) || 1);
    const ttlSec = Math.max(30, Math.min(600, Number(body.ttl_sec) || 180));
    if (!provider || !holderId) {
      return json({ ok: false, error: "provider and holder_id required" }, 400);
    }
    const slots = await loadSlots(env);
    const now = Date.now() / 1000;
    const list = (slots[provider] || []).filter((s) => s.until > now && s.holder_id !== holderId);
    if (list.length >= maxInflight) {
      return json({ ok: false, held: list.length, max: maxInflight });
    }
    list.push({ holder_id: holderId, until: now + ttlSec });
    slots[provider] = list;
    await saveSlots(env, slots);
    return json({ ok: true, held: list.length, max: maxInflight });
  }

  if (url.pathname === "/ledger/slots/release" && request.method === "POST") {
    const body = await request.json();
    const provider = String(body.provider || "");
    const holderId = String(body.holder_id || "");
    const slots = await loadSlots(env);
    const now = Date.now() / 1000;
    slots[provider] = (slots[provider] || []).filter(
      (s) => s.until > now && s.holder_id !== holderId,
    );
    await saveSlots(env, slots);
    return json({ ok: true });
  }

  // Delayed one-shot review retries after capacity defer.
  if (url.pathname === "/ledger/retries" && request.method === "POST") {
    const body = await request.json();
    const retries = await loadRetries(env);
    const repo = body.repo;
    const pr = Number(body.pr_number);
    // supersede prior pending for same PR
    for (const r of retries) {
      if (r.repo === repo && Number(r.pr_number) === pr && r.status === "pending") {
        r.status = "cancelled";
      }
    }
    retries.push({
      retry_id: crypto.randomUUID().slice(0, 12),
      repo,
      pr_number: pr,
      installation_id: body.installation_id,
      comment_id: body.comment_id || null,
      not_before: Number(body.not_before) || Date.now() / 1000 + 90,
      status: "pending",
      created_at: Date.now() / 1000,
    });
    await saveRetries(env, retries);
    return json({ ok: true, pending: retries.filter((r) => r.status === "pending").length });
  }

  return new Response("Not found", { status: 404 });
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export default {
  async fetch(request, env) {
    try {
      const url = new URL(request.url);

      if (url.pathname === "/health") {
        return new Response("ok", { status: 200 });
      }

      if (url.pathname.startsWith("/ledger/")) {
        return handleLedger(request, env, url);
      }

      if (url.pathname !== "/webhook" || request.method !== "POST") {
        return new Response("Not found", { status: 404 });
      }

      const body = await request.text();
      const signature = request.headers.get("X-Hub-Signature-256") || "";

      const valid = await verifySignature(body, signature, env.GITHUB_WEBHOOK_SECRET);
      if (!valid) {
        return new Response("Invalid signature", { status: 401 });
      }

      const event = request.headers.get("X-GitHub-Event") || "";
      const payload = JSON.parse(body);

      if (event === "ping") {
        return new Response(JSON.stringify({ ok: true, message: "pong" }), {
          headers: { "Content-Type": "application/json" },
        });
      }

      let result;
      if (event === "issue_comment") {
        result = await handleIssueComment(payload, env);
      } else {
        result = { skipped: true, event, reason: "only /sorge slash commands trigger review" };
      }

      return new Response(JSON.stringify(result), {
        status: result.error ? 500 : 200,
        headers: { "Content-Type": "application/json" },
      });
    } catch (err) {
      console.error("Unhandled error in fetch handler:", err.stack || err.message || err);
      return new Response(JSON.stringify({ error: String(err.message || err) }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      });
    }
  },

  async scheduled(event, env, ctx) {
    // Every minute: fire due auto-retries. Hourly (:20): escalate drain backup.
    const cron = event.cron || "";
    try {
      const flushed = await flushDueRetries(env);
      if (flushed.dispatched) {
        console.log(`scheduled retries: dispatched ${flushed.dispatched}`);
      }
    } catch (err) {
      console.error("flushDueRetries failed:", err.message || err);
    }

    const isHourlyDrain = cron === "20 * * * *";
    if (!isHourlyDrain) {
      return;
    }

    const installId = env.SORGE_DRAIN_INSTALLATION_ID || env.SORGE_DEFAULT_INSTALLATION_ID;
    if (!installId) {
      console.log("scheduled drain skipped: no SORGE_DRAIN_INSTALLATION_ID");
      return;
    }
    if (env.SORGE_LEDGER) {
      const data = await loadLedger(env);
      const pending = (data.tickets || []).filter((t) => t.status === "pending").length;
      if (pending === 0) {
        console.log("scheduled drain skipped: no pending tickets");
        return;
      }
      console.log(`scheduled drain: ${pending} pending ticket(s)`);
    }
    const result = await dispatchDrain(env, installId);
    console.log("scheduled drain result", JSON.stringify(result));
  },
};
