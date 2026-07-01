# Sorge GitHub App + Cloudflare Worker

Install Sorge on any org/repo **without adding workflow YAML** to consumer repositories.

## Architecture

```
PR event → GitHub App webhook → Cloudflare Worker → repository_dispatch → dispatch-review.yml → bot.main
```

## One-time setup

### 1. Create GitHub App

In GitHub org settings → Developer settings → GitHub Apps:

- **Webhook URL:** `https://sorge.<your-subdomain>.workers.dev/webhook`
- **Webhook secret:** generate and save
- **Permissions:** Pull requests (read), Contents (read), Issues (write), Metadata (read)
- **Events:** `Pull request`, `Installation`, `Installation repositories`

Save **App ID** and download **private key**.

### 2. Secrets in `deepiri-sorge` repository

| Secret | Description |
|--------|-------------|
| `SORGE_APP_ID` | GitHub App ID |
| `SORGE_APP_PRIVATE_KEY` | PEM private key (full contents) |
| `GOOGLE_API_KEY` | Gemini |
| `OPENROUTER_API_KEY` | Gemma |
| `GROQ_API_KEY` | Qwen/Groq |
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
