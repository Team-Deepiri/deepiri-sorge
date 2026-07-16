# Sorge Setup Guide

## Quick Comparison

| | Local Development | Production (GitHub App) |
| |---|-------------------|----------------------------|
| | GitHub Token | Your PAT or `GITHUB_TOKEN` env | App installation token (auto) |
| | API Key | Set in `.env` | Set in GitHub Secrets |
| | Config | `sorge.toml` | `sorge.toml` in repo |

---

## Option 1: GitHub App (Recommended — no YAML in consumer repos)

This is the primary way to use Sorge on external repositories. Reviews are triggered on-demand by commenting `/sorge` on a PR.

1. Install the **Sorge GitHub App** on your org or repository.
2. Comment `/sorge` on any PR you want reviewed.
3. Optionally add `sorge.toml` in the repo root for custom filters and routing.

No workflow files or API keys needed in your repo. See [docs/GITHUB_APP.md](GITHUB_APP.md) for full setup details.

---

## Option 2: Manual Dispatch (testing/debugging)

Manually trigger a review from the `deepiri-sorge` repo using `workflow_dispatch`.
All production reviews should go through the GitHub App / `/sorge` pipeline (Option 1).

### 1. Add API Key Secrets

Sorge supports multiple LLM providers. Add at least one:

| Secret | Provider | Get Key |
|--------|----------|---------|
| `GOOGLE_API_KEY` | Gemini (large PRs) | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `OPENROUTER_API_KEY` | OpenRouter (medium PRs) | [OpenRouter](https://openrouter.ai/keys) |
| `GROQ_API_KEY` | Groq (small PRs) | [Groq Console](https://console.groq.com/keys) |

1. Go to your repository on GitHub → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** for each key you want to add

### 2. (Optional) Add sorge.toml

Copy `sorge.toml` to your repo root if you want custom config. Defaults work fine out of the box.

---

## Option 3: Local Development

This is for testing the bot locally before deploying.

### 1. Get Your API Keys

**GitHub Token (for API access / diff fetching):**
1. Go to GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Fine-grained tokens**
2. Create new token with repo scope
3. Copy the token

**API Keys (at least one provider):**
- **Gemini**: [Google AI Studio](https://aistudio.google.com/app/apikey)
- **OpenRouter**: [OpenRouter Keys](https://openrouter.ai/keys)
- **Groq**: [Groq Console](https://console.groq.com/keys)

### 2. Set Up .env File

```bash
# Copy the example
cp .env.example .env

# Edit .env and fill in your keys:
# GITHUB_TOKEN=your_github_pat
# GOOGLE_API_KEY=your_gemini_key
# OPENROUTER_API_KEY=your_openrouter_key
# GROQ_API_KEY=your_groq_key
```

### 3. Run a Test

```bash
# Get a diff from any repo
git diff > test.diff

# Run the bot
python -m bot.main --diff test.diff --verbose
```

Optional flags:
- `--pr-number 1` - PR number for commenting
- `--repo "owner/repo"` - Repo for commenting
- `--token "$GITHUB_TOKEN"` - Your GitHub token (for posting comments)
- `--dry-run` - Don't post comments, just print output

### Environment setup

1. Install Poetry (if you haven't)
```bash
# After installing, add to your PATH
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies
```bash
# from pyproject.toml via poetry
poetry install
```
This creates a virtual environment and installs all dependencies

---

## Provider Limits & Scheduling

Providers are **execution backends** for the `ReviewScheduler`. The scheduler picks among healthy providers using a market score (health, latency, RPM remaining, context fit, historical quality). Size thresholds in `[routing]` still influence chunk budgets; they are no longer a hard "Groq then OpenRouter then Gemini" failover chain.

| Provider | Daily Limit | Context Window | Notes |
|----------|-------------|----------------|-------|
| Groq (GPT OSS 120B) | 1000 req/day | 8K | Fast; small context |
| OpenRouter (free models) | 50 req/day | large | Medium/large chunks |
| Gemini 2.5 Flash | 20 req/day | 1M | Large context |

Provider outcome history is stored in `~/.cache/sorge/provider_stats.json` (EMA by size×language) and restored across Actions runs via cache.

---

## Configuration

Edit `sorge.toml` to customize:

```toml
[sorge]
enabled = true

[filters]
min_lines = 20
skip_docs = true
skip_deps = true
skip_tests = false

[review]
style = "concise"
include_security = true
include_performance = true

[gemini]
enabled = true
model = "gemini-2.5-flash"

[openrouter]
enabled = true

[groq]
enabled = true
model = "openai/gpt-oss-120b"

[routing]
small_pr_threshold = 5000
medium_pr_threshold = 200000
large_pr_threshold = 200000

[scheduler]
wall_clock_sec = 720
max_workers = 4
health_threshold = 25.0
partial_on_exhausted = true

[providers.groq]
rpm = 30
max_inflight = 1
max_context_tokens = 8000
quality_prior = 0.9
nominal_latency_ms = 400

[providers.openrouter]
rpm = 20
max_inflight = 1
max_context_tokens = 100000
quality_prior = 0.7
nominal_latency_ms = 800

[providers.gemini]
rpm = 10
max_inflight = 1
max_context_tokens = 200000
quality_prior = 0.85
nominal_latency_ms = 1200
```

Defaults are tuned for free-tier RPM; raise `rpm` / `max_inflight` only if your keys allow it.
