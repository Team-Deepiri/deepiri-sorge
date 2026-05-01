# Sorge Setup Guide

## Quick Comparison

| | Local Development | Production (GitHub Actions) |
|---|-------------------|----------------------------|
| GitHub Token | Your PAT | Auto-available (`secrets.GITHUB_TOKEN`) |
| Gemini Key | Set in `.env` | Set in GitHub Secrets |
| Config | `sorge.toml` | `sorge.toml` in repo |

---

## Option 1: Production Setup (GitHub Actions)

This is what you want for your repo's PRs to be auto-reviewed.

### 1. Add Gemini API Key Secret

1. Go to your repository on GitHub
2. Go to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `GOOGLE_API_KEY`
5. Value: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
6. Click **Add secret**

### 2. (Optional) Add sorge.toml

Copy `sorge.toml` to your repo root if you want custom config. Defaults work fine out of the box.

### 3. Done!

The workflow automatically runs on PRs. No other setup needed.

---

## Option 2: Local Development

This is for testing the bot locally before deploying.

### 1. Get Your API Keys

**GitHub Token (for GitHub Models):**
1. Go to GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Fine-grained tokens**
2. Create new token:
   - **Name**: Sorge Dev
   - **Repository access**: Select your test repo(s)
   - **Permissions**: Add "Models" → "Read and write"
3. Copy the token

**Gemini API Key:**
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create or use existing API key
3. Copy the key

### 2. Set Up .env File

```bash
# Copy the example
cp .env.example .env

# Edit .env and fill in your keys:
# GITHUB_TOKEN=your_github_pat
# GOOGLE_API_KEY=your_gemini_key
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

---

## Model Limits

| Model | Daily Limit | Context | Best For |
|-------|-------------|---------|----------|
| GitHub Models (GPT-4o) | 150 requests | 128K | Small/medium PRs (<10K tokens) |
| Gemini 2.5 Pro | 100 requests | 1M | Large PRs (>25K tokens) |

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

[github_models]
enabled = true         # Enable GitHub Models (default: on)
model = "gpt-4o"       # Model to use

[gemini]
enabled = true          # Enable Gemini (default: on)
model = "gemini-2.5-pro-preview-0506"

[routing]
small_pr_threshold = 10000   # Use GitHub Models for diffs under this many tokens
large_pr_threshold = 25000   # Use Gemini for diffs over this many tokens
```

Defaults are tuned for typical PRs, so you likely don't need to change anything.