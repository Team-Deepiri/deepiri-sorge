# deepiri-sorge

**Distributed, event-driven AI PR review bot - runs on GitHub Actions for $0-10/month**

```
GitHub PR Event → GitHub Action (FREE CPU) → Decision Engine → [CPU Quantized Model | Skip | GPU Fallback] → PR Comment
```

## The Problem

Traditional AI code review bots require:
- Always-on GPU servers ($$$)
- Complex infrastructure
- Centralized backend
- Monthly costs well above $10

## The Solution: deepiri-sorge

A GitHub-native, distributed AI review bot that:

1. **Runs entirely in GitHub Actions** - Free compute for most PRs
2. **Uses quantized CPU models** - 7B models that run on standard runners
3. **Filters aggressively** - Skips 70-90% of PRs that don't need AI
4. **Optional GPU fallback** - Only for large/complex diffs

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GitHub PR Event                       │
│                  (push / PR open / update)                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   GitHub Action Runner                        │
│               (per-repo CPU execution)                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│    Decision Engine       │     │    Diff Pre-filtering        │
│  (rules + complexity)    │     │ (trivial/docs/small PRs)     │
└───────────┬─────────────┘     └──────────────┬──────────────┘
            │                                   │
            │ small PR                          │ skip
            ▼                                   ▼
┌─────────────────────────┐         ┌───────────────────────────┐
│   Quantized CPU Model   │         │     Skip / Light Comment  │
│       (7B Q4/Q5)        │         └───────────────────────────┘
└───────────┬─────────────┘
            │
            │ complex / large PR
            ▼
┌─────────────────────────────────────────────────────────────┐
│              GPU Runner (Optional)                           │
│         (RunPod / Vast.ai / Spot Instance)                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Structured Review Output                        │
│           (summary, issues, recommendations)                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    GitHub PR Comment                         │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Zero-cost default**: Uses GitHub Actions free minutes for CPU inference
- **Quantized models**: 7B models via llama.cpp, runs on 4-6GB RAM
- **Smart filtering**: Skips docs-only, small, trivial PRs
- **Distributed**: Each repo runs independently - no central server
- **Reusable workflows**: One workflow template, deploy to 20+ repos
- **GPU fallback**: Optional serverless GPU for complex diffs
- **Configurable**: Per-repo settings via `sorge.toml`
- **Cost estimation**: Built-in cost tracking

## Quick Start

### 1. Add to your repo

Copy [`.github/workflows/consumer-pr-review.example.yml`](.github/workflows/consumer-pr-review.example.yml) to your repo as `.github/workflows/pr_review.yml` and **pin a release tag** (not `@main`):

```yaml
jobs:
  review:
    uses: Team-Deepiri/deepiri-sorge/.github/workflows/sorge-review.yml@v0.1.0
    with:
      bot_ref: v0.1.0
    secrets:
      GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
      OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
      GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
```

Or run via Docker after a release:

```bash
docker run --rm -e GITHUB_TOKEN -e GROQ_API_KEY ghcr.io/team-deepiri/deepiri-sorge:v0.1.0 \
  --diff pr.diff --repo owner/repo --pr-number 1
```

### 2. Configure (optional)

Create `sorge.toml` in your repo root:

```toml
[sorge]
enabled = true

[filters]
min_lines = 20
skip_docs = true
skip_deps = true
max_cpu_lines = 500

[review]
style = "concise"  # concise | detailed | minimal
languages = ["python", "javascript", "typescript"]

[gpu]
enabled = false
threshold_lines = 1000
endpoint = ""  # Your RunPod/Vast.ai endpoint
```

## Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download quantized model
python -m bot.download_model

# Run locally
python -m bot.main --diff tests/fixtures/sample.diff

# Run tests
pytest tests/
```

### Docker

```bash
# CPU version (free tier)
docker build -f docker/Dockerfile.cpu -t sorge-cpu .
docker run sorge-cpu --diff pr.diff

# GPU version
docker build -f docker/Dockerfile.gpu -t sorge-gpu .
docker run --gpus all sorge-gpu --diff pr.diff
```

## How It Works

1. **PR Event Triggered** → GitHub Actions runs on `pull_request` events
2. **Diff Extraction** → Fetches and formats the PR diff
3. **Decision Engine** → Rules-based filtering:
   - Lines < 20 → Skip
   - Docs-only → Skip  
   - Dependencies → Skip
   - Small PRs → CPU review
   - Large PRs → GPU fallback (if enabled)
4. **AI Review** → Runs quantized model or calls GPU endpoint
5. **Post Comment** → Formats and posts review to PR

## Cost Breakdown

| Scenario | Monthly Cost |
|----------|-------------|
| 100 small PRs (CPU) | $0 |
| 50 medium PRs (CPU) | $0 |
| 10 large PRs (GPU) | ~$5-8 |
| 20 repos, moderate usage | **$0-10** |

## Project Structure

```
deepiri-sorge/
├── .github/workflows/      # GitHub Actions
├── bot/                    # Core bot code
│   ├── main.py            # Entry point
│   ├── decision_engine.py # PR filtering
│   ├── cpu_reviewer.py    # Quantized model runner
│   ├── gpu_runner.py      # Optional GPU endpoint
│   ├── comment_poster.py  # GitHub API integration
│   └── prompts/           # Review templates
├── docker/                # Container builds
├── tests/                 # Test suite
└── docs/                  # Documentation
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
