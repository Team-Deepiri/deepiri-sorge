"""Context shaving — Gemini-dead fallback only."""

from __future__ import annotations

from bot.context_shaver import (
    content_sha,
    gemini_fully_dead,
    layer0_shaving,
    pool_shavings,
    should_engage_context_shave,
)
from bot.diff_parser import ParsedDiff
from bot.file_splitter import ReviewChunk
from bot.quota_tracker import QuotaTracker


class _P:
    def __init__(self, name: str):
        self.name = name


def _chunk(files: list[str], tokens: int, raw: str | None = None) -> ReviewChunk:
    body = raw or ("+def foo():\n+    return 1\n" * max(1, tokens // 20))
    return ReviewChunk(
        files=files,
        parsed_diff=ParsedDiff(
            raw=body,
            files=files,
            lines_added=body.count("\n"),
            lines_deleted=0,
            files_changed=len(files),
        ),
        estimated_tokens=tokens,
    )


def test_gemini_fully_dead_when_quota_exhausted():
    quota = QuotaTracker(limits={"gemini": 1, "gpt": 10, "openrouter": 10}, used={"gemini": 1, "gpt": 0, "openrouter": 0})
    assert gemini_fully_dead(quota, [_P("gemini"), _P("groq")]) is True


def test_gemini_not_dead_when_quota_remains():
    quota = QuotaTracker(limits={"gemini": 20, "gpt": 10, "openrouter": 10}, used={"gemini": 0, "gpt": 0, "openrouter": 0})
    assert gemini_fully_dead(quota, [_P("gemini"), _P("groq")]) is False


def test_should_not_engage_when_gemini_usable_even_if_oversize():
    quota = QuotaTracker(limits={"gemini": 20, "gpt": 10, "openrouter": 10}, used={"gemini": 0, "gpt": 0, "openrouter": 0})
    chunks = [_chunk(["a.py"], 50_000)]
    engage, reason = should_engage_context_shave(
        enabled=True,
        quota=quota,
        providers=[_P("gemini"), _P("groq")],
        chunks=chunks,
        prompt_overhead=2500,
    )
    assert engage is False
    assert reason == "gemini_usable"


def test_should_not_engage_when_flag_off():
    quota = QuotaTracker(limits={"gemini": 1, "gpt": 10, "openrouter": 10}, used={"gemini": 1, "gpt": 0, "openrouter": 0})
    chunks = [_chunk(["a.py"], 50_000)]
    engage, reason = should_engage_context_shave(
        enabled=False,
        quota=quota,
        providers=[_P("gemini"), _P("groq")],
        chunks=chunks,
        prompt_overhead=2500,
    )
    assert engage is False
    assert reason == "flag_off"


def test_should_engage_when_gemini_dead_and_oversize():
    quota = QuotaTracker(limits={"gemini": 1, "gpt": 10, "openrouter": 10}, used={"gemini": 1, "gpt": 0, "openrouter": 0})
    chunks = [_chunk(["cli/agent/AgentWorker.js"], 20_000)]
    engage, reason = should_engage_context_shave(
        enabled=True,
        quota=quota,
        providers=[_P("gemini"), _P("groq")],
        chunks=chunks,
        prompt_overhead=2500,
    )
    assert engage is True
    assert reason == "gemini_dead_oversize"


def test_should_not_engage_when_gemini_dead_but_fits_groq():
    quota = QuotaTracker(limits={"gemini": 1, "gpt": 10, "openrouter": 10}, used={"gemini": 1, "gpt": 0, "openrouter": 0})
    # Diff tokens small enough that fit_diff = 7000-2500 = 4500 still covers it
    chunks = [_chunk(["a.py"], 800)]
    engage, reason = should_engage_context_shave(
        enabled=True,
        quota=quota,
        providers=[_P("gemini"), _P("groq")],
        chunks=chunks,
        prompt_overhead=2500,
    )
    assert engage is False
    assert reason == "fits_secondary"


def test_layer0_shaving_extracts_exports_and_is_compact():
    raw = "\n".join(
        [
            "+++ b/cli/agent/queue.js",
            "@@",
            "+async function runQueue(jobs) {",
            "+  const x = await fetch(url);",
            "+}",
            "+import { foo } from './bar';",
        ]
    )
    chunk = _chunk(["cli/agent/queue.js"], 200, raw=raw)
    sh = layer0_shaving(chunk, slice_id="s1")
    assert sh.content_sha == content_sha(raw)
    assert "runQueue" in sh.exports
    assert "fetch" in sh.side_effects
    pooled = pool_shavings([sh], max_chars=2000)
    assert "CONTEXT_SHAVINGS" in pooled
    assert "runQueue" in pooled


def test_shaving_cache_roundtrip(tmp_path, monkeypatch):
    from bot.utils import shaving_cache

    monkeypatch.setattr(shaving_cache, "CACHE_DIR", tmp_path)
    payload = {"slice_id": "s", "files": ["a.py"], "exports": ["f"], "imports": [], "side_effects": [], "fingerprints": [], "notes": "", "source": "heuristic", "content_sha": "abc"}
    shaving_cache.set("abc", payload)
    got = shaving_cache.get("abc", ttl_hours=24)
    assert got is not None
    assert got["exports"] == ["f"]
