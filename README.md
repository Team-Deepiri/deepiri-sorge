# deepiri-sorge

**On-demand AI PR review bot for GitHub — powered by Gemini, OpenRouter, and Groq**

```
PR @sorge mention → Cloudflare Worker → repository_dispatch → Decision Engine → [Skip | Groq | OpenRouter | Gemini] → PR Comment
```

## The Problem

Traditional AI code review bots require:
- Always-on GPU servers ($$$)
- Complex infrastructure
- Centralized backend
- Monthly costs well above $10

## The Solution: deepiri-sorge

A GitHub-native, on-demand AI review bot that:

1. **Runs in GitHub Actions** — Zero infrastructure to manage
2. **Routes to optimal LLM** — Groq for small PRs, OpenRouter for medium, Gemini for large
3. **Filters aggressively** — Skips 70-90% of PRs that don't need AI (when running automatic mode)
4. **Provider failover** — Falls back through the provider chain if quotas are exhausted

### Architecture

```
┌─────────────────────────────────────────────┐
│          PR comment "@sorge review"           │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│         Cloudflare Worker                    │
│   (checks for @sorge, dispatches review)     │
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

- **On-demand reviews**: Triggered by commenting `@sorge` on a PR — no auto-run on every push
- **Multiple LLM providers**: Routes PRs to Groq, OpenRouter, or Gemini based on size
- **Automatic failover**: Falls through the provider chain if quotas are exhausted
- **Smart filtering**: Skips docs-only, dependency-only, test-only, and trivial PRs (overridable with `@sorge`)
- **Configurable routing**: Per-repo thresholds via `sorge.toml` and environment variables
- **GitHub App support**: Webhook-based dispatch via Cloudflare Worker — no YAML in consumer repos
- **Provider-agnostic**: Swap providers or add new ones via the runner interface
- **Cost tracking**: Built-in quota management per provider

## Quick Start

### Recommended: GitHub App (zero YAML in consumer repos)

1. Install the **Sorge GitHub App** on your org or repository.
2. Comment `@sorge` on a PR when you want a review (on-demand only — no auto-run on every push).
3. Optionally add `sorge.toml` in the repo root for filters and routing.

No workflow files, no API keys in consumer repos — see [docs/GITHUB_APP.md](docs/GITHUB_APP.md).

### Self-hosted

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
small_pr_threshold = 5000   # Groq for small PRs
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

## How It Works

1. **`@sorge` Comment** → Someone comments `@sorge` on a PR; the Cloudflare Worker detects the mention
2. **Dispatch** → Worker sends a `repository_dispatch` to the central `deepiri-sorge` repo
3. **Diff Extraction** → Fetches and parses the PR diff
4. **Filtering** → Decision engine applies rules: skip docs-only, deps-only, test-only, or below min_lines (overridden when triggered via `@sorge`)
5. **Routing** → Routes the diff to the optimal provider based on token estimate
6. **Review** → Provider (Groq / OpenRouter / Gemini) generates structured review
7. **Fallback** → If the primary provider fails or quota is exhausted, falls through the preference chain
8. **Post Comment** → Formats and posts the review to the PR

## Cost Breakdown

All providers are used via their free tiers. No paid API usage is required.

| Provider | Free Tier Limit |
|----------|----------------|
| Groq | 1000 requests / day |
| OpenRouter | 50 requests / day |
| Gemini 2.5 Flash | 20 requests / day |

**Monthly cost: $0** — no GPU servers, no inference infrastructure, no paid API keys needed.

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