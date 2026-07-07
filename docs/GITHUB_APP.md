# Sorge GitHub App + Cloudflare Worker

Install Sorge on any org/repo **without adding workflow YAML** to consumer repositories.

## Architecture

```
PR comment /sorge → GitHub App webhook → Cloudflare Worker → repository_dispatch → dispatch-review.yml → bot.main
```

Reviews run **only when someone uses /sorge** on a PR comment — not automatically on every push.

## One-time setup

### 1. Create GitHub App

In GitHub org settings → Developer settings → GitHub Apps:

- **Webhook URL:** `https://sorge.<your-subdomain>.workers.dev/webhook`
- **Webhook secret:** generate and save
- **Permissions:** Pull requests (read), Contents (read), Issues (write), Metadata (read)
- **Events:** `Issue comment`, `Installation`, `Installation repositories`

  (`Issue comment` covers PR comments; Sorge ignores comments that do not contain `/sorge`.)

Save **App ID** and download **private key**.

### 2. Secrets in `deepiri-sorge` repository

| Secret | Description |
|--------|-------------|
| `SORGE_APP_ID` | GitHub App ID |
| `SORGE_APP_PRIVATE_KEY` | PEM private key (full contents) |
| `GOOGLE_API_KEY` | Gemini |
| `OPENROUTER_API_KEY` | Gemma/OpenRouter |
| `GROQ_API_KEY` | GPT/Groq |
| `GITHUB_DISPATCH_TOKEN` | PAT with `repo` scope (for Worker dispatch) |

### 3. Deploy Cloudflare Worker

```bash
cd worker
npx wrangler secret put GITHUB_WEBHOOK_SECRET
npx wrangler secret put GITHUB_DISPATCH_TOKEN
npx wrangler deploy
```

### 4. Install the App

Install the GitHub App on target org/repos. No YAML required in those repos.

## Manual dispatch (testing)

```bash
gh workflow run dispatch-review.yml \
  -f repo=Team-Deepiri/some-repo \
  -f pr_number=42 \
  -f installation_id=12345678
```

Install the GitHub App on target org/repos. No per-repo workflow YAML is required.

## On-demand review (`/sorge`)

Comment on any PR:

```text
/sorge
```

The GitHub App receives an `issue_comment` webhook, the Worker checks for `/sorge`, and dispatches the same central review workflow. Slash-command-triggered runs pass `--force` so small/docs-only filters do not skip the review.

If your App bot login is not `sorge`, set `SORGE_BOT_LOGIN` on the Worker (e.g. `deepiri-sorge`).
