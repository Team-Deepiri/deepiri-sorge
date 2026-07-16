# deepiri-sorge

**On-demand AI PR review bot for GitHub — powered by Gemini, OpenRouter, and Groq**
---
```
PR /sorge → Cloudflare Worker → repository_dispatch → filters → ReviewScheduler → providers → PR Comment
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
2. **Schedules providers as backends** — market-scored pick among Groq / OpenRouter / Gemini (no worker-owned fallback stampede)
3. **Smart filtering** — Skips docs-only, dependency-only, and trivial PRs when not forced
4. **Priority + partial review** — security/auth chunks first; low-priority may skip under rate limits or deadline

### Architecture

```
┌─────────────────────────────────────────────┐
│          PR comment "/sorge"                   │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│         Cloudflare Worker                    │
│   (checks for /sorge, dispatches review)     │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│     Filters (DecisionEngine) + FileSplitter  │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│              ReviewScheduler                 │
│  priority queue · token buckets · health     │
│  market score · chunk cache · history EMA    │
└───────┬─────────────────┬────────────┬──────┘
        ▼                 ▼            ▼
   ┌──────────┐  ┌────────────┐  ┌──────────┐
   │   Groq   │  │ OpenRouter │  │  Gemini  │
   └──────────┘  └────────────┘  └──────────┘
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

- **On-demand reviews**: Triggered by commenting `/sorge` on a PR — no auto-run on every push
- **Provider-centric scheduler**: RPM token buckets, continuous health, adaptive concurrency
- **Market scoring**: health · latency · RPM · context fit · historical quality (5%)
- **Path priority**: auth/security before docs/lockfiles under pressure
- **Scheduler chunk cache**: skip duplicate provider calls for the same diff+context
- **Smart filtering**: Skips docs-only, dependency-only, test-only, and trivial PRs (overridable with `/sorge`)
- **Configurable**: Per-repo thresholds via `sorge.toml` and environment variables
- **GitHub App support**: Webhook-based dispatch via Cloudflare Worker — no YAML in consumer repos
- **Cost tracking**: Built-in quota management per provider

## Quick Start

### Recommended: GitHub App (zero YAML in consumer repos)

1. Install the **Sorge GitHub App** on your org or repository.
2. Comment `/sorge` on a PR when you want a review (on-demand only — no auto-run on every push).
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
small_pr_threshold = 5000
medium_pr_threshold = 200000
large_pr_threshold = 200000

[scheduler]
wall_clock_sec = 720
max_workers = 4
health_threshold = 25.0

[providers.groq]
rpm = 30
max_inflight = 1
max_context_tokens = 8000

[providers.openrouter]
rpm = 20
max_inflight = 1
max_context_tokens = 100000

[providers.gemini]
rpm = 10
max_inflight = 1
max_context_tokens = 200000
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

1. **`/sorge` Command** → Someone comments `/sorge` on a PR; the Cloudflare Worker detects the slash command
2. **Dispatch** → Worker sends a `repository_dispatch` to the central `deepiri-sorge` repo
3. **Diff Extraction** → Fetches and parses the PR diff (checks out **PR head**)
4. **Filtering** → Decision engine applies rules: skip docs-only, deps-only, test-only, or below min_lines (overridden when triggered via `/sorge`)
5. **Schedule** → FileSplitter chunks the diff; scheduler prioritizes paths and picks a provider via market score
6. **Review** → One provider call per dispatch (adapters wrap Groq / OpenRouter / Gemini runners)
7. **Partial under pressure** → On 429/deadline, high-priority chunks finish; others may be skipped with an explicit note
8. **Post Comment** → Formats and posts the review to the PR (routing details include scheduler meta)

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
│   ├── decision_engine.py  # PR filtering
│   ├── scheduling/         # ReviewScheduler platform
│   ├── providers/          # Groq / OpenRouter / Gemini adapters
│   ├── comment_poster.py   # GitHub API integration
│   ├── runners/            # LLM HTTP runners
```

See [docs/SETUP.md](docs/SETUP.md) for secrets and local setup.
