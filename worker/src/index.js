/**
 * Sorge webhook dispatcher — receives GitHub App pull_request events
 * and triggers the central review workflow via repository_dispatch.
 *
 * Env vars:
 *   GITHUB_WEBHOOK_SECRET    — HMAC secret for verifying webhook payloads
 *   GITHUB_DISPATCH_TOKEN    — PAT or App token with repo scope for dispatch API
 *   SORGE_DISPATCH_REPO      — Target repo for repository_dispatch (default: Team-Deepiri/deepiri-sorge)
 *
 * Deploy: npx wrangler deploy
 */

const DISPATCH_EVENT = "sorge-review";

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

async function dispatchReview(env, payload) {
  const DISPATCH_REPO = env.SORGE_DISPATCH_REPO || "Team-Deepiri/deepiri-sorge";
  const pr = payload.pull_request;
  const repo = payload.repository.full_name;
  const installationId = payload.installation?.id;

  if (!pr || !installationId) {
    return { skipped: true, reason: "missing pr or installation" };
  }

  if (pr.head?.repo?.full_name !== pr.base?.repo?.full_name) {
    return { skipped: true, reason: "fork PR — same-repo only by default" };
  }

  const token = env.GITHUB_DISPATCH_TOKEN;
  if (!token) {
    return { error: "GITHUB_DISPATCH_TOKEN not configured" };
  }

  const res = await fetch(
    `https://api.github.com/repos/${DISPATCH_REPO}/dispatches`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        event_type: DISPATCH_EVENT,
        client_payload: {
          repo,
          pr_number: pr.number,
          installation_id: installationId,
          action: payload.action,
        },
      }),
    },
  );

  if (!res.ok) {
    const text = await res.text();
    return { error: `dispatch failed: ${res.status} ${text}` };
  }

  return { ok: true, repo, pr_number: pr.number };
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return new Response("ok", { status: 200 });
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

    if (event !== "pull_request") {
      return new Response(JSON.stringify({ skipped: true, event }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    const action = payload.action;
    if (!["opened", "synchronize", "reopened"].includes(action)) {
      return new Response(JSON.stringify({ skipped: true, action }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    const result = await dispatchReview(env, payload);
    return new Response(JSON.stringify(result), {
      status: result.error ? 500 : 200,
      headers: { "Content-Type": "application/json" },
    });
  },
};
