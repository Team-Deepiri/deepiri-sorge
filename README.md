# deepiri-sorge

**Distributed, event-driven AI PR review bot for GitHub — powered by Gemini, OpenRouter, and Groq**

```
GitHub PR Event → GitHub Action → Decision Engine → [Skip | Groq | OpenRouter | Gemini] → PR Comment
```

## The Problem

Traditional AI code review bots require:
- Always-on GPU servers ($$$)
- Complex infrastructure
- Centralized backend
- Monthly costs well above $10

## The Solution: deepiri-sorge

A GitHub-native, distributed AI review bot that:

1. **Runs in GitHub Actions** — Zero infrastructure to manage
2. **Routes to optimal LLM** — Groq for small PRs, OpenRouter for medium, Gemini for large
3. **Filters aggressively** — Skips 70-90% of PRs that don't need AI
4. **Provider failover** — Falls back through the provider chain if quotas are exhausted

### Architecture

```
┌─────────────────────────────────────────────┐
│              GitHub PR Event                 │
│         (opened / synchronize / reopened)    │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│         GitHub Actions Dispatch               │
│     (via Cloudflare Worker webhook proxy)     │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│           Decision Engine                    │
│  (line count, file types, security scan)     │
└────────┬──────────────┬──────────────┬───────┘
         │              │              │
         ▼              ▼              ▼
   ┌──────────┐  ┌────────────┐  ┌──────────┐
   │   Groq   │  │ OpenRouter │  │  Gemini  │
   │ (small)  │  │ (medium)   │  │ (large)  │
   └──────────┘  └────────────┘  └──────────┘
         │              │              │
         └──────────────┴──────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│          Structured Review Output             │
│      (summary, issues, recommendations)       │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│               GitHub PR Comment              │
└─────────────────────────────────────────────┘
```

## Features

- **Multiple LLM providers**: Routes PRs to Groq, OpenRouter, or Gemini based on size
- **Automatic failover**: Falls through the provider chain if quotas are exhausted
- **Smart filtering**: Skips docs-only, dependency-only, test-only, and trivial PRs
- **Configurable routing**: Per-repo thresholds via `sorge.toml` and environment variables
- **GitHub App support**: Optional webhook-based dispatch via Cloudflare Worker
- **Provider-agnostic**: Swap providers or add new ones via the runner interface
- **Cost tracking**: Built-in quota management per provider

## Quick Start

### Recommended: GitHub App (zero YAML in consumer repos)

1. Install the **Sorge GitHub App** on your org or repository.
2. Optionally add `sorge.toml` in the repo root for filters and routing.

No workflow files, no API keys in consumer repos — see [docs/GITHUB_APP.md](docs/GITHUB_APP.md).

No workflow files or API keys are required in consumer repos — see [docs/GITHUB_APP.md](docs/GITHUB_APP.md).

```bash
docker run --rm -e GITHUB_TOKEN -e GROQ_API_KEY ghcr.io/team-deepiri/deepiri-sorge:v0.1.0 \
  --diff pr.diff --repo owner/repo --pr-number 1
```

Create `sorge.toml` in your repo root:

```toml
[sorge]
enabled = true

[filters]
min_lines = 20
skip_docs = true
skip_deps = true
skip_tests = false

[review]
style = "concise"  # concise | detailed | minimal
include_security = true
include_performance = true

[routing]
small_pr_threshold = 3700   # Groq for small PRs
medium_pr_threshold = 200000 # OpenRouter for medium PRs
large_pr_threshold = 200000  # Gemini for large PRs
```

## Usage

### Local Development

```bash
# Install dependencies
poetry install

# Run locally (requires GOOGLE_API_KEY or OPENROUTER_API_KEY or GROQ_API_KEY in .env)
python -m bot.main --diff tests/fixtures/sample.diff

# Run tests
pytest tests/
```

### Docker

```bash
docker build -f docker/Dockerfile.cpu -t sorge-cpu .
docker run sorge-cpu --diff pr.diff
```

## How It Works

1. **PR Event** → GitHub Actions workflow or Cloudflare Worker webhook dispatches the review
2. **Diff Extraction** → Fetches and parses the PR diff
3. **Filtering** → Decision engine applies rules: skip docs-only, deps-only, test-only, or below min_lines
4. **Routing** → Routes the diff to the optimal provider based on token estimate
5. **Review** → Provider (Groq / OpenRouter / Gemini) generates structured review
6. **Fallback** → If the primary provider fails or quota is exhausted, falls through the preference chain
7. **Post Comment** → Formats and posts the review to the PR

## Cost Breakdown

| Scenario | Monthly Cost |
|----------|-------------|
| 100 small PRs (Groq free tier) | $0 |
| 50 medium PRs (OpenRouter) | ~$2-5 |
| 10 large PRs (Gemini free tier) | $0 |
| Mixed usage, all providers | **$0-10** |

## Project Structure

```
deepiri-sorge/
├── .github/workflows/       # GitHub Actions workflows
├── bot/                     # Core bot code
│   ├── main.py             # Entry point & dispatch
│   ├── config.py           # Configuration (Pydantic)
│   ├── decision_engine.py  # PR filtering & routing
│   ├── context_router.py   # Token-aware provider routing
│   ├── comment_poster.py   # GitHub API integration
│   ├── runners/            # LLM provider runners
│   │   ├── gemini_runner.py
│   │   ├── openrouter_runner.py
│   │   └── groq_runner.py
│   └── prompts/            # Review templates
├── worker/                 # Cloudflare Worker (webhook dispatch)
├── docker/                 # Container builds
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## Releases (CD)

Pushing a semver tag triggers [`.github/workflows/cd.yml`](.github/workflows/cd.yml):

1. Runs the test suite
2. Creates a [GitHub Release](https://github.com/Team-Deepiri/deepiri-sorge/releases)
3. Publishes `ghcr.io/team-deepiri/deepiri-sorge:<tag>`

```bash
git tag v0.2.0
git push origin v0.2.0
```

Consumer workflows should pin the same tag for `uses:` and `bot_ref:`.

## License

Copyright 2026 Deepiri. Licensed under Apache License 2.0.
