# Sorge Setup Guide

## Quick Comparison

| | Local Development | Production (GitHub Actions) |
|---|-------------------|----------------------------|
| GitHub Token | Your PAT or `GITHUB_TOKEN` env | Auto-available |
| API Key | Set in `.env` | Set in GitHub Secrets |
| Config | `sorge.toml` | `sorge.toml` in repo |

---

## Option 1: Production Setup (GitHub Actions)

This is what you want for your repo's PRs to be auto-reviewed.

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

### 3. Done!

The workflow automatically runs on PRs. No other setup needed.

---

## Option 2: Local Development

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

## Provider Limits & Routing

| Provider | Daily Limit | Context Window | Best For |
|----------|-------------|----------------|----------|
| Groq (GPT OSS 120B) | 1000 req/day | 8K | Small PRs (<5000 tokens) |
| OpenRouter (Gemma 4) | 50 req/day | 1M | Medium PRs (5000–200K tokens) |
| Gemini 2.5 Flash | 20 req/day | 1M | Large PRs (>200K tokens) |

---

## Configuration

Edit `sorge.toml` to customize:

```toml
[sorge]
enabled = true

[filters]
min_lines = 20          # Skip PRs smaller than this
skip_docs = true        # Skip docs-only PRs
skip_deps = true        # Skip dependency-only PRs
skip_tests = false      # Skip test-only PRs

[review]
style = "concise"       # concise | detailed | minimal
include_security = true
include_performance = true

[gemini]
enabled = true          # Enable Gemini (default: on)
model = "gemini-2.5-flash"

[openrouter]
enabled = true          # Enable OpenRouter (default: on)
model = "google/gemma-4-31b-it:free"

[groq]
enabled = true          # Enable Groq (default: on)
model = "openai/gpt-oss-120b"

[routing]
small_pr_threshold = 5000     # Groq for diffs under this many tokens
medium_pr_threshold = 200000  # OpenRouter for diffs under this many tokens
large_pr_threshold = 200000   # Gemini for diffs over this many tokens
```

Defaults are tuned for typical PRs, so you likely don't need to change anything.