/**
 * Sorge webhook dispatcher — triggers review only when @sorge is mentioned
 * on a PR comment (issue_comment event).
 *
 * Env vars:
 *   GITHUB_WEBHOOK_SECRET    — HMAC secret for verifying webhook payloads
 *   GITHUB_DISPATCH_TOKEN    — PAT or App token with repo scope for dispatch API
 *   SORGE_DISPATCH_REPO      — Target repo for repository_dispatch (default: Team-Deepiri/deepiri-sorge)
 *   SORGE_BOT_LOGIN          — Extra mention handle if bot login is not "sorge"
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

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** True when a PR comment body @-mentions Sorge. */
export function mentionsSorge(body, extraLogin) {
  if (!body || typeof body !== "string") return false;

  const handles = ["sorge"];
  if (extraLogin) {
    handles.push(extraLogin.replace(/^@/, ""));
  }

  return handles.some((handle) => {
    const escaped = escapeRegex(handle);
    // @sorge, @sorge[bot], @sorge-ai, etc.
    const pattern = new RegExp(`@${escaped}(-[\\w-]+)?(\\[bot\\])?\\b`, "i");
    return pattern.test(body);
  });
}

async function dispatchReview(env, { repo, prNumber, installationId, trigger }) {
  if (!repo || !prNumber || !installationId) {
    return { skipped: true, reason: "missing repo, pr_number, or installation" };
  }

  const token = env.GITHUB_DISPATCH_TOKEN;
  if (!token) {
    return { error: "GITHUB_DISPATCH_TOKEN not configured" };
  }

  const dispatchRepo = env.SORGE_DISPATCH_REPO || "Team-Deepiri/deepiri-sorge";

  const res = await fetch(
    `https://api.github.com/repos/${dispatchRepo}/dispatches`,
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
          pr_number: prNumber,
          installation_id: installationId,
          trigger: trigger || "mention",
          force: trigger === "mention",
        },
      }),
    },
  );

  if (!res.ok) {
    const text = await res.text();
    return { error: `dispatch failed: ${res.status} ${text}` };
  }

  return { ok: true, repo, pr_number: prNumber, trigger: trigger || "mention" };
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

  if (!mentionsSorge(comment.body, env.SORGE_BOT_LOGIN)) {
    return { skipped: true, reason: "no @sorge mention" };
  }

  return dispatchReview(env, {
    repo: payload.repository.full_name,
    prNumber: issue.number,
    installationId: payload.installation?.id,
    trigger: "mention",
  });
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

    let result;
    if (event === "issue_comment") {
      result = await handleIssueComment(payload, env);
    } else {
      result = { skipped: true, event, reason: "only @sorge mentions trigger review" };
    }

    return new Response(JSON.stringify(result), {
      status: result.error ? 500 : 200,
      headers: { "Content-Type": "application/json" },
    });
  },
};
