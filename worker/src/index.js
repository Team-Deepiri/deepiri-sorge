/**
 * Sorge webhook dispatcher — triggers review only when /sorge slash command is used
 * on a PR comment (issue_comment event).
 *
 * Env vars:
 *   GITHUB_WEBHOOK_SECRET    — HMAC secret for verifying webhook payloads
 *   SORGE_APP_ID             — GitHub App ID (same one used by the Actions workflow / bot)
 *   SORGE_APP_PRIVATE_KEY    — GitHub App private key, PKCS#8 PEM format (see note below)
 *   SORGE_DISPATCH_REPO      — Target repo for repository_dispatch (default: Team-Deepiri/deepiri-sorge)
 *   SORGE_BOT_LOGIN          — Extra command handle if bot login is not "sorge"
 *
 * Deploy: npx wrangler deploy
 *
 * NOTE on private key format:
 *   GitHub issues App private keys in PKCS#1 format (-----BEGIN RSA PRIVATE KEY-----).
 *   The Web Crypto API (used here, since Workers don't have Node's crypto module)
 *   requires PKCS#8 (-----BEGIN PRIVATE KEY-----). Convert once with:
 *
 *     openssl pkcs8 -topk8 -nocrypt -in original-app-key.pem -out app-key-pkcs8.pem
 *
 *   Then store the *contents* of app-key-pkcs8.pem as the SORGE_APP_PRIVATE_KEY secret:
 *
 *     npx wrangler secret put SORGE_APP_PRIVATE_KEY
 *     (paste the full contents of app-key-pkcs8.pem, including header/footer lines)
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

/** True when a PR comment body contains a /sorge slash command. */
export function hasSorgeSlashCommand(body, extraLogin) {
  if (!body || typeof body !== "string") return false;

  const handles = ["sorge"];
  if (extraLogin) {
    handles.push(extraLogin.replace(/^\//, "").replace(/^@/, ""));
  }

  return handles.some((handle) => {
    const escaped = escapeRegex(handle);
    // /sorge, /Sorge, /sorge-ai, etc.
    const pattern = new RegExp(`/${escaped}(-[\\w-]+)?\\b`, "i");
    return pattern.test(body);
  });
}

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

/** Signs a GitHub App JWT (RS256) using the App's PKCS#8 private key. */
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

/** Exchanges the App JWT for a short-lived installation token, scoped to one repo. */
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

async function dispatchReview(env, { repo, prNumber, installationId, trigger }) {
  if (!repo || !prNumber || !installationId) {
    return { skipped: true, reason: "missing repo, pr_number, or installation" };
  }

  if (!env.SORGE_APP_ID || !env.SORGE_APP_PRIVATE_KEY) {
    return { error: "SORGE_APP_ID or SORGE_APP_PRIVATE_KEY not configured" };
  }

  const dispatchRepo = env.SORGE_DISPATCH_REPO || "Team-Deepiri/deepiri-sorge";
  const dispatchRepoName = dispatchRepo.split("/").pop();

  let token;
  try {
    token = await getInstallationToken(env, installationId, dispatchRepoName);
  } catch (err) {
    return { error: `failed to mint installation token: ${err.message || err}` };
  }

  const res = await fetch(
    `https://api.github.com/repos/${dispatchRepo}/dispatches`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "sorge-webhook-worker",
      },
      body: JSON.stringify({
        event_type: DISPATCH_EVENT,
        client_payload: {
          repo,
          pr_number: prNumber,
          installation_id: installationId,
          trigger: trigger || "slash_command",
          force: trigger === "slash_command",
        },
      }),
    },
  );

  if (!res.ok) {
    const text = await res.text();
    return { error: `dispatch failed: ${res.status} ${text}` };
  }

  return { ok: true, repo, pr_number: prNumber, trigger: trigger || "slash_command" };
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

export default {
  async fetch(request, env) {
    try {
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
        result = { skipped: true, event, reason: "only /sorge slash commands trigger review" };
      }

      return new Response(JSON.stringify(result), {
        status: result.error ? 500 : 200,
        headers: { "Content-Type": "application/json" },
      });
    } catch (err) {
      console.error("Unhandled error in fetch handler:", err.stack || err.message || err);
      return new Response(
        JSON.stringify({ error: String(err.message || err) }),
        {
          status: 500,
          headers: { "Content-Type": "application/json" },
        },
      );
    }
  },
};