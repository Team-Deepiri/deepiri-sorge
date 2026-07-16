"""Tests for provider history EMA store."""

from pathlib import Path

from bot.diff_parser import ParsedDiff
from bot.file_splitter import ReviewChunk
from bot.scheduling.history import ProviderHistory, language_for_chunk, size_bucket
from bot.scheduling.market_score import score_provider
from bot.scheduling.types import ProviderStatus, ScheduledChunk


def _chunk(files: list[str], tokens: int = 100) -> ReviewChunk:
    return ReviewChunk(
        files=files,
        parsed_diff=ParsedDiff(raw="+x\n"),
        estimated_tokens=tokens,
    )


def test_size_and_language_helpers():
    assert size_bucket(100) == "small"
    assert size_bucket(20_000) == "medium"
    assert language_for_chunk(_chunk(["src/a.py", "b.py"])) == "py"
    assert language_for_chunk(_chunk(["readme.md"])) == "md"


def test_history_persist_reload(tmp_path: Path):
    path = tmp_path / "provider_stats.json"
    h = ProviderHistory(path, sync_remote=False)
    chunk = _chunk(["app/main.py"], tokens=2000)
    h.record("groq", chunk, ok=True, latency_ms=100)
    h.record("groq", chunk, ok=True, latency_ms=200)
    h.save()

    h2 = ProviderHistory(path, sync_remote=False)
    q = h2.quality("groq", chunk, default=0.1)
    assert q > 0.5


def test_history_cross_run_cooldown(tmp_path: Path):
    path = tmp_path / "provider_stats.json"
    h = ProviderHistory(path, sync_remote=False)
    h.mark_rate_limited("openrouter", retry_after=60)
    h.save()
    h2 = ProviderHistory(path, sync_remote=False)
    assert h2.is_cooling("openrouter")
    assert h2.cooling_remaining("openrouter") > 30


def test_history_review_score_ema(tmp_path: Path):
    path = tmp_path / "provider_stats.json"
    h = ProviderHistory(path, sync_remote=False)
    chunk = _chunk(["src/util.py"], tokens=1000)
    h.record("groq", chunk, ok=True, latency_ms=100, review_score=6.0, issues_found=3)
    h.record("groq", chunk, ok=True, latency_ms=120, review_score=8.0, issues_found=2)
    q = h.quality("groq", chunk, default=0.1)
    assert q > 0.5
    h.save()
    h2 = ProviderHistory(path, sync_remote=False)
    assert h2.quality("groq", chunk, default=0.1) > 0.5
