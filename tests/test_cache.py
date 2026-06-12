"""Tests for bot/utils/cache.py"""

from __future__ import annotations

import json
import time

import pytest

from tests.helpers import install_loguru_stub

install_loguru_stub()

from bot.utils import cache as _cache


DIFF = "diff --git a/foo.py b/foo.py\n+print('hello')"
MODEL = "gemini-2.5-flash"
RESULT = {"summary": "Looks good", "issues": [], "recommendations": [], "score": 9.0}


@pytest.fixture(autouse=True)
def clean_cache(tmp_path, monkeypatch):
    """Redirect cache dir to a temp directory for each test."""
    monkeypatch.setattr(_cache, "CACHE_DIR", tmp_path)
    yield
    # cleanup handled by tmp_path


class TestCacheSetAndGet:
    def test_set_then_get_returns_result(self):
        _cache.set(DIFF, MODEL, RESULT)
        cached = _cache.get(DIFF, MODEL, ttl_hours=24)
        assert cached == RESULT

    def test_get_missing_key_returns_none(self):
        assert _cache.get("nonexistent diff", MODEL, ttl_hours=24) is None

    def test_different_models_are_separate_entries(self):
        _cache.set(DIFF, "model-a", {"score": 1.0})
        _cache.set(DIFF, "model-b", {"score": 2.0})
        assert _cache.get(DIFF, "model-a")["score"] == 1.0
        assert _cache.get(DIFF, "model-b")["score"] == 2.0

    def test_different_diffs_are_separate_entries(self):
        _cache.set("diff-a", MODEL, {"score": 1.0})
        _cache.set("diff-b", MODEL, {"score": 2.0})
        assert _cache.get("diff-a", MODEL)["score"] == 1.0
        assert _cache.get("diff-b", MODEL)["score"] == 2.0


class TestCacheTTL:
    def test_expired_entry_returns_none(self, tmp_path):
        _cache.set(DIFF, MODEL, RESULT)

        # Manually backdate the cached_at timestamp
        path = _cache._cache_path(DIFF, MODEL)
        data = json.loads(path.read_text())
        data["cached_at"] = time.time() - 25 * 3600  # 25 hours ago
        path.write_text(json.dumps(data))

        assert _cache.get(DIFF, MODEL, ttl_hours=24) is None

    def test_expired_entry_is_deleted(self, tmp_path):
        _cache.set(DIFF, MODEL, RESULT)

        path = _cache._cache_path(DIFF, MODEL)
        data = json.loads(path.read_text())
        data["cached_at"] = time.time() - 25 * 3600
        path.write_text(json.dumps(data))

        _cache.get(DIFF, MODEL, ttl_hours=24)

        assert not path.exists()

    def test_fresh_entry_within_ttl_is_returned(self):
        _cache.set(DIFF, MODEL, RESULT)
        assert _cache.get(DIFF, MODEL, ttl_hours=1) == RESULT


class TestCacheInvalidateAndClear:
    def test_invalidate_removes_entry(self):
        _cache.set(DIFF, MODEL, RESULT)
        _cache.invalidate(DIFF, MODEL)
        assert _cache.get(DIFF, MODEL) is None

    def test_invalidate_nonexistent_does_not_raise(self):
        _cache.invalidate("does not exist", MODEL)  # should not raise

    def test_clear_all_removes_all_entries(self):
        _cache.set("diff-1", MODEL, RESULT)
        _cache.set("diff-2", MODEL, RESULT)
        count = _cache.clear_all()
        assert count == 2
        assert _cache.get("diff-1", MODEL) is None
        assert _cache.get("diff-2", MODEL) is None

    def test_clear_all_empty_cache_returns_zero(self):
        assert _cache.clear_all() == 0