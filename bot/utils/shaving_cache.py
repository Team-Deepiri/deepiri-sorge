"""SHA-keyed cache for context shavings (invalidate on content hash miss)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger

CACHE_DIR = Path.home() / ".cache" / "sorge" / "shavings"


def get(content_sha: str, ttl_hours: int = 168) -> dict | None:
    """Return cached shaving dict if present and fresh."""
    path = CACHE_DIR / f"{content_sha}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug(f"Shaving cache read failed: {exc}")
        return None
    age_h = (time.time() - float(data.get("cached_at", 0))) / 3600.0
    if age_h > ttl_hours:
        return None
    shaving = data.get("shaving")
    return shaving if isinstance(shaving, dict) else None


def set(content_sha: str, shaving: dict) -> None:
    """Persist a shaving keyed by content SHA."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{content_sha}.json"
    payload = {"cached_at": time.time(), "shaving": shaving}
    try:
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    except OSError as exc:
        logger.debug(f"Shaving cache write failed: {exc}")
