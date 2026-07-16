"""In-memory + cross-run API quota tracking (soft RPD continuity)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from bot.config import QuotaConfig

DEFAULT_QUOTA_PATH = Path.home() / ".cache" / "sorge" / "quota_daily.json"


@dataclass
class QuotaTracker:
    """Track successful API calls against hardcoded RPD limits.

    Seeds from env, local quota_daily.json, and optional Worker KV ledger
    (`/ledger/quota`) so concurrent Actions runs soft-share free-tier budgets.
    """

    limits: dict[str, int] = field(default_factory=dict)
    used: dict[str, int] = field(default_factory=dict)
    warn_at_pct: float = 0.8
    adjustments: list[str] = field(default_factory=list)
    persist_path: Path | None = None
    sync_remote: bool = True

    PROVIDER_KEYS = {
        "groq": "gpt",
        "openrouter": "openrouter",
        "gemini": "gemini",
    }

    @classmethod
    def from_config(
        cls,
        config: QuotaConfig,
        *,
        persist_path: Path | None = None,
        sync_remote: bool = True,
    ) -> QuotaTracker:
        limits = {
            "gemini": config.gemini_rpd,
            "gpt": config.gpt_rpd,
            "openrouter": config.openrouter_rpd,
        }
        used = {k: 0 for k in limits}
        path = persist_path or Path(
            os.getenv("SORGE_QUOTA_FILE") or str(DEFAULT_QUOTA_PATH)
        )
        file_used = cls._load_file_used(path)
        for key, val in file_used.items():
            if key in used:
                used[key] = max(used[key], val)
        for env_name, key in (
            ("SORGE_GEMINI_USED_TODAY", "gemini"),
            ("SORGE_GPT_USED_TODAY", "gpt"),
            ("SORGE_OPENROUTER_USED_TODAY", "openrouter"),
        ):
            if os.getenv(env_name):
                try:
                    used[key] = max(used[key], int(os.getenv(env_name, "0")))
                except ValueError:
                    pass
        tracker = cls(
            limits=limits,
            used=used,
            warn_at_pct=config.warn_at_pct,
            persist_path=path,
            sync_remote=sync_remote,
        )
        if sync_remote:
            tracker._pull_remote()
        return tracker

    def _pull_remote(self) -> None:
        try:
            from bot.escalate_ledger import EscalateLedger

            remote = EscalateLedger().fetch_quota_used()
            for key, val in remote.items():
                if key in self.used:
                    self.used[key] = max(self.used[key], val)
        except Exception as e:
            logger.debug(f"Quota remote pull skipped: {e}")

    def _push_remote(self) -> None:
        if not self.sync_remote:
            return
        try:
            from bot.escalate_ledger import EscalateLedger

            EscalateLedger().push_quota_used(dict(self.used))
        except Exception as e:
            logger.debug(f"Quota remote push skipped: {e}")

    @staticmethod
    def _utc_today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    @classmethod
    def _load_file_used(cls, path: Path) -> dict[str, int]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Quota file unreadable ({path}): {e}")
            return {}
        if data.get("date") != cls._utc_today():
            return {}
        used = data.get("used") or {}
        out: dict[str, int] = {}
        for k, v in used.items():
            try:
                out[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
        return out

    def save(self) -> None:
        if self.persist_path is not None:
            try:
                self.persist_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "date": self._utc_today(),
                    "used": dict(self.used),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                self.persist_path.write_text(json.dumps(payload, indent=2))
            except OSError as e:
                logger.warning(f"Quota file save failed: {e}")
        self._push_remote()

    def remaining(self, provider: str) -> int:
        key = self.PROVIDER_KEYS.get(provider, provider)
        return max(0, self.limits.get(key, 0) - self.used.get(key, 0))

    def can_use(self, provider: str) -> bool:
        """True when remaining > soft reserve (keep 1 free-tier call as buffer)."""
        key = self.PROVIDER_KEYS.get(provider, provider)
        remaining = self.remaining(provider)
        soft_reserve = 1 if key in ("gemini", "openrouter") else 0
        return remaining > soft_reserve

    def record(self, provider: str) -> None:
        key = self.PROVIDER_KEYS.get(provider, provider)
        self.used[key] = self.used.get(key, 0) + 1
        limit = self.limits.get(key, 0)
        if limit and self.used[key] / limit >= self.warn_at_pct:
            self.adjustments.append(
                f"{key} at {self.used[key]}/{limit} ({self.used[key] / limit:.0%})"
            )
        self.save()

    def record_failure(self, provider: str) -> None:
        """Soft-count rate-limit failures so the scheduler avoids hot providers."""
        key = self.PROVIDER_KEYS.get(provider, provider)
        self.used[key] = self.used.get(key, 0) + 1
        self.adjustments.append(f"{key} rate-limited this run")
        self.save()

    def snapshot(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for key, limit in self.limits.items():
            used = self.used.get(key, 0)
            out[key] = {"limit": limit, "used": used, "remaining": max(0, limit - used)}
        return out
