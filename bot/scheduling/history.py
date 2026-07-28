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
# Longer default cool so free-tier storms don't get pecked across Actions runs.
DEFAULT_COOLDOWN_SEC = 180.0
MAX_COOLDOWN_SEC = 600.0

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
    """Thread-safe EMA success/latency/review-score store with JSON persistence.

    Cooldowns are also max-merged through Worker KV when SORGE_LEDGER_* is set,
    so concurrent Actions runs see each other's 429s instead of stampeding.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        alpha: float = EMA_ALPHA,
        sync_remote: bool = True,
    ):
        self.path = path or DEFAULT_PATH
        self.alpha = alpha
        self.sync_remote = sync_remote
        self._lock = threading.Lock()
        self._stats: dict[str, dict] = {}
        self._cooldowns: dict[str, float] = {}  # provider -> unix ts until
        self.load()
        if sync_remote:
            self._pull_remote_cooldowns()

    def _pull_remote_cooldowns(self) -> None:
        try:
            from bot.escalate_ledger import EscalateLedger

            remote = EscalateLedger().fetch_provider_cooldowns()
            if not remote:
                return
            now = time.time()
            with self._lock:
                for name, until in remote.items():
                    if until <= now:
                        continue
                    prev = self._cooldowns.get(name, 0.0)
                    self._cooldowns[name] = max(prev, until)
        except Exception as e:
            logger.debug(f"Provider cooldown remote pull skipped: {e}")

    def _push_remote_cooldowns(self) -> None:
        if not self.sync_remote:
            return
        try:
            from bot.escalate_ledger import EscalateLedger

            with self._lock:
                cool = dict(self._cooldowns)
            EscalateLedger().push_provider_cooldowns(cool)
        except Exception as e:
            logger.debug(f"Provider cooldown remote push skipped: {e}")

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
                now = time.time()
                cool = {k: v for k, v in self._cooldowns.items() if v > now}
                self._cooldowns = cool
                payload = {"version": 3, "stats": self._stats, "cooldowns": cool}
                self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            except Exception as exc:
                logger.warning(f"provider history save failed: {exc}")
        self._push_remote_cooldowns()

    def mark_rate_limited(
        self,
        provider: str,
        *,
        retry_after: float | None = None,
    ) -> None:
        wait = retry_after if retry_after is not None else DEFAULT_COOLDOWN_SEC
        wait = min(max(float(wait), 30.0), MAX_COOLDOWN_SEC)
        until = time.time() + wait
        with self._lock:
            prev = self._cooldowns.get(provider, 0.0)
            self._cooldowns[provider] = max(prev, until)
        logger.info(f"Provider history: {provider} cooled for {wait:.0f}s (cross-run)")
        self._push_remote_cooldowns()

    def cooling_remaining(self, provider: str) -> float:
        with self._lock:
            return max(0.0, self._cooldowns.get(provider, 0.0) - time.time())

    def is_cooling(self, provider: str) -> bool:
        return self.cooling_remaining(provider) > 0.0

    def max_cooling_remaining(self) -> float:
        with self._lock:
            now = time.time()
            if not self._cooldowns:
                return 0.0
            return max(0.0, max(self._cooldowns.values()) - now)

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
            # Review usefulness (0–10 → 0–1) when present
            review_q = row.get("ema_review_score")
            if review_q is not None:
                score_n = max(0.0, min(1.0, float(review_q) / 10.0))
                return max(
                    0.0,
                    min(1.0, 0.50 * success + 0.20 * lat_factor + 0.30 * score_n),
                )
            return max(0.0, min(1.0, 0.75 * success + 0.25 * lat_factor))

    def record(
        self,
        provider: str,
        chunk: ReviewChunk,
        *,
        ok: bool,
        latency_ms: float = 0.0,
        review_score: float | None = None,
        issues_found: int | None = None,
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
                row = self._stats[key]
                if ok and review_score is not None:
                    row["ema_review_score"] = float(review_score)
                if ok and issues_found is not None:
                    row["ema_issues"] = float(issues_found)
            else:
                a = self.alpha
                row["ema_success"] = a * success + (1 - a) * float(row.get("ema_success", 0.5))
                if ok and latency_ms > 0:
                    prev = float(row.get("ema_latency_ms", latency_ms))
                    row["ema_latency_ms"] = a * latency_ms + (1 - a) * prev
                if ok and review_score is not None:
                    prev_s = float(row.get("ema_review_score", review_score))
                    row["ema_review_score"] = a * float(review_score) + (1 - a) * prev_s
                if ok and issues_found is not None:
                    prev_i = float(row.get("ema_issues", issues_found))
                    row["ema_issues"] = a * float(issues_found) + (1 - a) * prev_i
                row["n"] = int(row.get("n", 0)) + 1
