"""Persistent provider outcome history (EMA) for market scoring.

Stores a small fixed-cardinality scoreboard under ~/.cache/sorge/provider_stats.json.
Keys are provider × size_bucket × language — never an unbounded event log.

Also tracks cross-run provider cooldowns after 429 so the next Actions job
does not immediately peck a hot free-tier endpoint.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from loguru import logger

from bot.file_splitter import ReviewChunk

DEFAULT_PATH = Path.home() / ".cache" / "sorge" / "provider_stats.json"
EMA_ALPHA = 0.3
DEFAULT_COOLDOWN_SEC = 90.0

_EXT_LANG = {
    ".py": "py",
    ".pyi": "py",
    ".js": "js",
    ".jsx": "js",
    ".ts": "ts",
    ".tsx": "ts",
    ".go": "go",
    ".rs": "rs",
    ".java": "java",
    ".kt": "kt",
    ".rb": "rb",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "cs",
    ".swift": "swift",
    ".md": "md",
    ".toml": "toml",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".sh": "sh",
    ".bash": "sh",
}


def size_bucket(tokens: int) -> str:
    if tokens <= 5_000:
        return "small"
    if tokens <= 50_000:
        return "medium"
    if tokens <= 200_000:
        return "large"
    return "xlarge"


def language_for_chunk(chunk: ReviewChunk) -> str:
    counts: dict[str, int] = {}
    for path in chunk.files:
        lower = path.lower()
        lang = "other"
        for ext, name in _EXT_LANG.items():
            if lower.endswith(ext):
                lang = name
                break
        counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return "other"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _stat_key(provider: str, bucket: str, lang: str) -> str:
    return f"{provider}|{bucket}|{lang}"


class ProviderHistory:
    """Thread-safe EMA success/latency store with JSON persistence."""

    def __init__(self, path: Path | None = None, *, alpha: float = EMA_ALPHA):
        self.path = path or DEFAULT_PATH
        self.alpha = alpha
        self._lock = threading.Lock()
        self._stats: dict[str, dict] = {}
        self._cooldowns: dict[str, float] = {}  # provider -> unix ts until
        self.load()

    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._stats = {}
                self._cooldowns = {}
                return
            try:
                data = json.loads(self.path.read_text())
                if not isinstance(data, dict):
                    self._stats = {}
                    self._cooldowns = {}
                    return
                if "stats" in data:
                    self._stats = data.get("stats") or {}
                    self._cooldowns = {
                        k: float(v) for k, v in (data.get("cooldowns") or {}).items()
                    }
                else:
                    # Legacy: whole file was the stats map
                    self._stats = data
                    self._cooldowns = {}
            except Exception as exc:
                logger.warning(f"provider history load failed: {exc}")
                self._stats = {}
                self._cooldowns = {}

    def save(self) -> None:
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # Drop expired cooldowns
                now = time.time()
                cool = {k: v for k, v in self._cooldowns.items() if v > now}
                self._cooldowns = cool
                payload = {"version": 2, "stats": self._stats, "cooldowns": cool}
                self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            except Exception as exc:
                logger.warning(f"provider history save failed: {exc}")

    def mark_rate_limited(
        self,
        provider: str,
        *,
        retry_after: float | None = None,
    ) -> None:
        wait = retry_after if retry_after is not None else DEFAULT_COOLDOWN_SEC
        wait = min(max(float(wait), 15.0), 300.0)
        until = time.time() + wait
        with self._lock:
            prev = self._cooldowns.get(provider, 0.0)
            self._cooldowns[provider] = max(prev, until)
        logger.info(f"Provider history: {provider} cooled for {wait:.0f}s (cross-run)")

    def cooling_remaining(self, provider: str) -> float:
        with self._lock:
            return max(0.0, self._cooldowns.get(provider, 0.0) - time.time())

    def is_cooling(self, provider: str) -> bool:
        return self.cooling_remaining(provider) > 0.0

    def quality(
        self,
        provider: str,
        chunk: ReviewChunk,
        *,
        default: float = 0.5,
    ) -> float:
        """Return historical quality in [0, 1] for market score."""
        key = _stat_key(provider, size_bucket(chunk.estimated_tokens), language_for_chunk(chunk))
        with self._lock:
            row = self._stats.get(key)
            if not row or row.get("n", 0) < 1:
                return default
            success = float(row.get("ema_success", default))
            lat = float(row.get("ema_latency_ms", 800.0))
            lat_factor = max(0.0, min(1.0, 1.0 - (lat / 4000.0)))
            return max(0.0, min(1.0, 0.75 * success + 0.25 * lat_factor))

    def record(
        self,
        provider: str,
        chunk: ReviewChunk,
        *,
        ok: bool,
        latency_ms: float = 0.0,
    ) -> None:
        key = _stat_key(provider, size_bucket(chunk.estimated_tokens), language_for_chunk(chunk))
        success = 1.0 if ok else 0.0
        with self._lock:
            row = self._stats.get(key)
            if not row:
                self._stats[key] = {
                    "ema_success": success,
                    "ema_latency_ms": float(latency_ms) if ok else 0.0,
                    "n": 1,
                }
            else:
                a = self.alpha
                row["ema_success"] = a * success + (1 - a) * float(row.get("ema_success", 0.5))
                if ok and latency_ms > 0:
                    prev = float(row.get("ema_latency_ms", latency_ms))
                    row["ema_latency_ms"] = a * latency_ms + (1 - a) * prev
                row["n"] = int(row.get("n", 0)) + 1
