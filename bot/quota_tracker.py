"""In-memory per-run API quota tracking."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from bot.config import QuotaConfig


@dataclass
class QuotaTracker:
    """Track successful API calls against hardcoded RPD limits for this run."""

    limits: dict[str, int] = field(default_factory=dict)
    used: dict[str, int] = field(default_factory=dict)
    warn_at_pct: float = 0.8
    adjustments: list[str] = field(default_factory=list)

    PROVIDER_KEYS = {
        "groq": "gpt",
        "openrouter": "openrouter",
        "gemini": "gemini",
    }

    @classmethod
    def from_config(cls, config: QuotaConfig) -> QuotaTracker:
        limits = {
            "gemini": config.gemini_rpd,
            "gpt": config.gpt_rpd,
            "openrouter": config.openrouter_rpd,
        }
        used = {k: 0 for k in limits}
        for env_name, key in (
            ("SORGE_GEMINI_USED_TODAY", "gemini"),
            ("SORGE_GPT_USED_TODAY", "gpt"),
            ("SORGE_OPENROUTER_USED_TODAY", "openrouter"),
        ):
            if os.getenv(env_name):
                try:
                    used[key] = int(os.getenv(env_name, "0"))
                except ValueError:
                    pass
        return cls(limits=limits, used=used, warn_at_pct=config.warn_at_pct)

    def remaining(self, provider: str) -> int:
        key = self.PROVIDER_KEYS.get(provider, provider)
        return max(0, self.limits.get(key, 0) - self.used.get(key, 0))

    def can_use(self, provider: str) -> bool:
        """True when remaining > soft reserve (keep 1 free-tier call as buffer)."""
        key = self.PROVIDER_KEYS.get(provider, provider)
        remaining = self.remaining(provider)
        # Soft daily budget: never spend the last free-tier slot on pecking.
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

    def record_failure(self, provider: str) -> None:
        """Soft-count rate-limit failures so the scheduler avoids hot providers."""
        key = self.PROVIDER_KEYS.get(provider, provider)
        # Count as half a request toward the limit — enough to steer away under storm.
        self.used[key] = self.used.get(key, 0) + 1
        self.adjustments.append(f"{key} rate-limited this run")

    def snapshot(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for key, limit in self.limits.items():
            used = self.used.get(key, 0)
            out[key] = {"limit": limit, "used": used, "remaining": max(0, limit - used)}
        return out