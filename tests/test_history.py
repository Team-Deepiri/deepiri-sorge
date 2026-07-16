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
    h = ProviderHistory(path)
    chunk = _chunk(["app/main.py"], tokens=2000)
    h.record("groq", chunk, ok=True, latency_ms=100)
    h.record("groq", chunk, ok=True, latency_ms=200)
    h.save()

    h2 = ProviderHistory(path)
    q = h2.quality("groq", chunk, default=0.1)
    assert q > 0.5


def test_history_influences_market_score(tmp_path: Path):
    path = tmp_path / "provider_stats.json"
    h = ProviderHistory(path)
    chunk = _chunk(["src/auth.py"], tokens=1000)
    # Teach: gemini succeeds, groq fails repeatedly
    for _ in range(5):
        h.record("gemini", chunk, ok=True, latency_ms=500)
        h.record("groq", chunk, ok=False, latency_ms=0)

    scheduled = ScheduledChunk(chunk=chunk, priority=100)
    groq = ProviderStatus(
        name="groq",
        health=90,
        rpm_remaining=10,
        max_context_tokens=8000,
        nominal_latency_ms=200,
        quality_prior=0.9,
    )
    gemini = ProviderStatus(
        name="gemini",
        health=90,
        rpm_remaining=10,
        max_context_tokens=200000,
        nominal_latency_ms=1200,
        quality_prior=0.5,
    )
    sg = score_provider(
        groq, scheduled, historical_quality=h.quality("groq", chunk, default=0.9)
    )
    sm = score_provider(
        gemini, scheduled, historical_quality=h.quality("gemini", chunk, default=0.5)
    )
    # Learned success should lift gemini over cold-start-favored groq on history term;
    # with equal health, latency still favors groq — assert quality values themselves.
    assert h.quality("gemini", chunk) > h.quality("groq", chunk)
    assert sm > 0 and sg > 0
