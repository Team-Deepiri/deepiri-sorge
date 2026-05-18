"""Disk-backed cache for review results, keyed by diff + model hash."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from loguru import logger

CACHE_DIR = Path.home() / ".cache" / "sorge" / "reviews"


def _cache_path(diff_raw: str, model: str) -> Path:
    key = hashlib.sha256(f"{model}:{diff_raw}".encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def get(diff_raw: str, model: str, ttl_hours: int = 24) -> dict | None:
    """Return cached result dict if present and not expired, else None."""
    path = _cache_path(diff_raw, model)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        age_hours = (time.time() - data["cached_at"]) / 3600
        if age_hours > ttl_hours:
            path.unlink(missing_ok=True)
            logger.debug(f"Cache expired ({age_hours:.1f}h old), evicted")
            return None
        logger.info(f"Cache hit ({age_hours:.1f}h old) for model={model}")
        return data["result"]
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
        return None


def set(diff_raw: str, model: str, result: dict) -> None:
    """Write result dict to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(diff_raw, model)
    try:
        path.write_text(json.dumps({"cached_at": time.time(), "result": result}))
        logger.debug(f"Cached result at {path.name}")
    except Exception as e:
        logger.warning(f"Cache write error: {e}")


def invalidate(diff_raw: str, model: str) -> None:
    """Delete a specific cache entry."""
    path = _cache_path(diff_raw, model)
    path.unlink(missing_ok=True)


def clear_all() -> int:
    """Delete all cached results. Returns number of entries removed."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for p in CACHE_DIR.glob("*.json"):
        p.unlink(missing_ok=True)
        count += 1
    logger.info(f"Cleared {count} cache entries")
    return count