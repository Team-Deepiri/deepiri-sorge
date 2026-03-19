# deepiri-sorge Implementation Plan

**Project:** Distributed AI PR Review Bot  
**Owner:** Deepiri  
**License:** Apache 2.0  
**Created:** 2026  
**Version:** 0.1.0

---

## Overview

deepiri-sorge is a distributed, event-driven AI code review bot that runs on GitHub Actions. It provides intelligent PR reviews at near-zero cost by leveraging:

- **GitHub Actions** for free CPU compute
- **Quantized models** (7B parameters) that run on standard runners
- **Smart filtering** to skip 70-90% of PRs that don't need AI review
- **Optional GPU fallback** for large/complex diffs

## Architecture

```
GitHub PR Event → GitHub Action → Decision Engine → [CPU Review | Skip | GPU Fallback] → PR Comment
```

### Components

| Component | Purpose | Location |
|-----------|--------|----------|
| Decision Engine | Filters PRs, determines review strategy | `bot/decision_engine.py` |
| Diff Parser | Parses and analyzes PR diffs | `bot/diff_parser.py` |
| CPU Reviewer | Runs quantized models on CPU | `bot/cpu_reviewer.py` |
| GPU Runner | Optional serverless GPU endpoint | `bot/gpu_runner.py` |
| Comment Poster | Posts reviews to GitHub PRs | `bot/comment_poster.py` |
| Config | Configuration management | `bot/config.py` |

---

## Implementation Status

### Phase 1: Core Infrastructure ✅

| Task | Status | Notes |
|------|--------|-------|
| Project structure | ✅ Complete | Full directory layout |
| LICENSE (Apache 2.0) | ✅ Complete | Deepiri 2026 |
| .gitignore | ✅ Complete | Comprehensive patterns |
| README.md | ✅ Complete | Documentation |
| requirements.txt | ✅ Complete | Dependencies |
| pyproject.toml | ✅ Complete | Packaging |

### Phase 2: Core Bot Module ✅

| Task | Status | Notes |
|------|--------|-------|
| `bot/__init__.py` | ✅ Complete | Package exports |
| `bot/main.py` | ✅ Complete | Entry point |
| `bot/config.py` | ✅ Complete | Configuration management |
| `bot/decision_engine.py` | ✅ Complete | PR filtering logic |
| `bot/diff_parser.py` | ✅ Complete | Diff analysis |
| `bot/cpu_reviewer.py` | ✅ Complete | CPU inference |
| `bot/gpu_runner.py` | ✅ Complete | GPU endpoint integration |
| `bot/comment_poster.py` | ✅ Complete | GitHub API |
| `bot/prompts/` | ✅ Complete | Review templates |

### Phase 3: GitHub Actions ✅

| Task | Status | Notes |
|------|--------|-------|
| `pr_review.yml` | ✅ Complete | Main workflow |
| `ci.yml` | ✅ Complete | CI/CD pipeline |

### Phase 4: Docker ✅

| Task | Status | Notes |
|------|--------|-------|
| `Dockerfile.cpu` | ✅ Complete | CPU image |
| `Dockerfile.gpu` | ✅ Complete | GPU image |

### Phase 5: Testing ✅

| Task | Status | Notes |
|------|--------|-------|
| `test_decision_engine.py` | ✅ Complete | Decision engine tests |
| `test_diff_parser.py` | ✅ Complete | Parser tests |
| `test_cpu_reviewer.py` | ✅ Complete | Reviewer tests |
| `test_config.py` | ✅ Complete | Config tests |
| `pytest.ini` | ✅ Complete | Test configuration |
| `tests/fixtures/` | ✅ Complete | Test data |

### Phase 6: Documentation 🔄

| Task | Status | Notes |
|------|--------|-------|
| `IMPLEMENTATION_PLAN.md` | ✅ Complete | This file |
| Architecture diagrams | ⏳ Pending | ASCII diagrams |
| API documentation | ⏳ Pending | Generated docs |
| Deployment guide | ⏳ Pending | Multi-repo setup |

---

## Features

### Completed

- [x] Event-driven PR review (on open/sync/reopen)
- [x] Diff extraction and parsing
- [x] Rule-based PR filtering (docs, deps, size)
- [x] Configurable via `sorge.toml`
- [x] Environment variable overrides
- [x] CPU inference with quantized models
- [x] Heuristic fallback when no model available
- [x] GitHub API integration for comments
- [x] Structured review output (JSON)
- [x] Configurable review styles
- [x] Multi-language support
- [x] Skip comment posting for filtered PRs

### In Progress

- [ ] llama.cpp integration for local inference
- [ ] Model download script
- [ ] Cache for review results
- [ ] PR complexity classifier (ML-based)

### Planned

- [ ] Batching for multiple small PRs
- [ ] Reusable workflow for multi-repo deployment
- [ ] Cost tracking and reporting
- [ ] Webhook configuration UI
- [ ] Dashboard for review statistics
- [ ] Custom rule engine
- [ ] Integration with code quality tools
- [ ] Support for GitHub Enterprise

WE NEED TO CONFIGURE IT TO GITHUB AND ANY EXTERNAL PLATFORM THAT WE USE OR EXTERNAL HOSTING OR MODEL
---

## Configuration

### sorge.toml

```toml
[sorge]
enabled = true

[filters]
min_lines = 20
skip_docs = true
skip_deps = true
skip_tests = false
max_cpu_lines = 500

[review]
style = "concise"
include_security = true
include_performance = true
include_style = true

[gpu]
enabled = false
threshold_lines = 1000
endpoint = ""
timeout = 60

[model]
name = "codellama-7b-q4"
context_size = 4096
threads = 4

[cache]
enabled = true
ttl_hours = 24
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SORGE_ENABLED` | Enable/disable bot | `true` |
| `SORGE_MIN_LINES` | Min lines for review | `20` |
| `SORGE_MAX_CPU_LINES` | Max lines for CPU | `500` |
| `SORGE_GPU_ENABLED` | Enable GPU fallback | `false` |
| `SORGE_GPU_ENDPOINT` | GPU endpoint URL | - |
| `SORGE_GPU_API_KEY` | GPU API key | - |
| `SORGE_MODEL_PATH` | Path to model | - |

---

## Deployment

### Single Repo

Add to `.github/workflows/pr_review.yml`:

```yaml
name: AI PR Review
on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  review:
    uses: deepiri/deepiri-sorge/.github/workflows/pr_review.yml@main
    with:
      pr_number: ${{ github.event.pull_request.number }}
    secrets:
      GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### Multi-Repo (Reusable Workflow)

1. Fork deepiri-sorge
2. Update workflow path in each repo
3. Optional: Create `sorge.toml` for repo-specific config

---

## Cost Analysis

| Scenario | Monthly Cost |
|----------|-------------|
| 100 small PRs (CPU) | $0 |
| 200 medium PRs (CPU) | $0 |
| 20 large PRs (GPU) | ~$5-8 |
| **Total (typical)** | **$0-10** |

### Cost Optimization

1. **Aggressive filtering** - Skip docs-only, deps, small PRs
2. **CPU-first** - Use quantized models on Actions runners
3. **GPU-only large** - Only trigger GPU for complex diffs
4. **Free Actions minutes** - Use GitHub's free tier

---

## File Structure

```
deepiri-sorge/
├── .github/workflows/
│   ├── pr_review.yml      # Main PR review workflow
│   └── ci.yml             # CI/CD pipeline
├── bot/
│   ├── __init__.py
│   ├── main.py            # Entry point
│   ├── config.py          # Configuration
│   ├── decision_engine.py # PR filtering
│   ├── diff_parser.py     # Diff analysis
│   ├── cpu_reviewer.py    # CPU inference
│   ├── gpu_runner.py      # GPU endpoints
│   ├── comment_poster.py  # GitHub API
│   ├── prompts/           # Review templates
│   │   ├── review_template.txt
│   │   └── skip_template.txt
│   ├── models/            # Model storage
│   └── utils/
│       ├── logging_utils.py
│       └── github_api.py
├── docker/
│   ├── Dockerfile.cpu
│   └── Dockerfile.gpu
├── tests/
│   ├── test_decision_engine.py
│   ├── test_diff_parser.py
│   ├── test_cpu_reviewer.py
│   ├── test_config.py
│   └── fixtures/
├── docs/
│   └── IMPLEMENTATION_PLAN.md
├── LICENSE
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## Next Steps

1. **Test locally** - Run `pytest tests/`
2. **Build Docker images** - `docker build -f docker/Dockerfile.cpu`
3. **Deploy to first repo** - Add workflow file
4. **Configure GPU** - Set up RunPod/Vast.ai endpoint (optional)
5. **Monitor costs** - Track GitHub Actions usage

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure CI passes
5. Submit a pull request

---

## License

Copyright 2026 Deepiri. Licensed under Apache License 2.0.
