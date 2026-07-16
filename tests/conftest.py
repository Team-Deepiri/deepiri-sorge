"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_sorge_cache(tmp_path, monkeypatch):
    """Keep scheduler/runner disk cache out of the developer's ~/.cache."""
    from bot.utils import cache as review_cache
    from bot.scheduling import history as hist

    monkeypatch.setattr(review_cache, "CACHE_DIR", tmp_path / "sorge-reviews")
    monkeypatch.setattr(hist, "DEFAULT_PATH", tmp_path / "provider_stats.json")
