"""Build enabled providers from config."""

from __future__ import annotations

from bot.config import CacheConfig, Config
from bot.providers.base import Provider
from bot.providers.gemini import GeminiProvider
from bot.providers.groq import GroqProvider
from bot.providers.openrouter import OpenRouterProvider


def build_providers(
    config: Config,
    *,
    cache_config: CacheConfig | None = None,
    only: str | None = None,
) -> list[Provider]:
    """Return enabled provider backends. `only` forces a single name (CLI --mode)."""
    providers: list[Provider] = []

    def want(name: str) -> bool:
        return only is None or only == name

    if want("groq") and config.groq.enabled:
        providers.append(GroqProvider(config, cache_config))
    if want("openrouter") and config.openrouter.enabled:
        providers.append(OpenRouterProvider(config, cache_config))
    if want("gemini") and config.gemini.enabled:
        providers.append(GeminiProvider(config, cache_config))

    return providers
