"""Configuration management for deepiri-sorge"""

import os
from pathlib import Path

import toml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class FiltersConfig(BaseModel):
    min_lines: int = Field(default=20, description="Minimum lines changed to trigger review")
    skip_docs: bool = Field(default=True, description="Skip docs-only changes")
    skip_deps: bool = Field(default=True, description="Skip dependency changes")
    skip_tests: bool = Field(default=False, description="Skip test-only changes")


class ReviewConfig(BaseModel):
    style: str = Field(default="concise", description="Review style: concise, detailed, minimal")
    languages: list[str] = Field(default_factory=lambda: ["*"], description="Languages to review")
    include_security: bool = Field(default=True, description="Include security checks")
    include_performance: bool = Field(default=True, description="Include performance checks")
    include_style: bool = Field(default=True, description="Include style checks")



class GeminiConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable Gemini")
    model: str = Field(default="gemini-2.5-flash", description="Model to use")
    api_key: str | None = Field(default=None, description="API key (uses GOOGLE_API_KEY env if not set)")


class OpenRouterConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable OpenRouter")
    model: str = Field(default="google/gemma-4-31b-it:free", description="Legacy: first model in the list")
    models: list[str] = Field(
        default=[
            "google/gemma-4-31b-it:free",
            "openai/gpt-oss-20b:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "google/gemma-4-26b-a4b-it:free",
        ],
        description="Ordered list of OpenRouter models to try in sequence (3 retries each)",
    )
    endpoint: str = Field(
        default="https://openrouter.ai/api/v1/chat/completions",
        description="OpenRouter chat completions endpoint",
    )
    api_key: str | None = Field(default=None, description="API key (uses OPENROUTER_API_KEY env if not set)")

    def model_post_init(self, __context):
        """After init, sync model from models[0] and vice versa."""
        fields_set = self.model_fields_set
        if "model" in fields_set and "models" not in fields_set:
            # Old-style config with only 'model' set; sync models from it
            self.models = [self.model]
        elif "models" in fields_set and "model" not in fields_set:
            # New-style config with only 'models' set; sync legacy field from it
            self.model = self.models[0]
        elif "models" in fields_set and "model" in fields_set:
            # Both set explicitly; ensure model matches models[0]
            self.model = self.models[0]
        # else: neither explicitly set (all defaults) — leave defaults untouched
        super().model_post_init(__context)


class GroqConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable Groq")
    model: str = Field(default="openai/gpt-oss-120b", description="Model to use")
    endpoint: str = Field(
        default="https://api.groq.com/openai/v1/chat/completions",
        description="Groq chat completions endpoint",
    )
    api_key: str | None = Field(default=None, description="API key (uses GROQ_API_KEY env if not set)")


class RoutingConfig(BaseModel):
    small_pr_threshold: int = Field(default=5000, description="Max tokens for small PR (Groq)")
    medium_pr_threshold: int = Field(default=120000, description="Max tokens for medium PR (OpenRouter)")
    large_pr_threshold: int = Field(default=120000, description="Min tokens for large PR (Gemini)")
    chunk_budget: int = Field(default=100000, description="Target tokens per chunk when splitting")
    context_shave_enabled: bool = Field(
        default=True,
        description=(
            "Gemini-dead fallback only: build a virtual context pool (shavings) "
            "so Groq/OpenRouter can review oversized PRs. Never engages while Gemini is usable."
        ),
    )
    context_shave_max_extracts: int = Field(
        default=12,
        description="Max LLM extract calls per run when Layer-0 shavings are thin",
    )
    context_shave_slice_budget: int = Field(
        default=3500,
        description="Diff tokens per shave slice (before prompt overhead)",
    )


class QuotaConfig(BaseModel):
    gemini_rpd: int = Field(default=20, description="Gemini requests per day")
    gpt_rpd: int = Field(default=1000, description="Groq/GPT OSS requests per day")
    openrouter_rpd: int = Field(default=50, description="OpenRouter requests per day")
    warn_at_pct: float = Field(default=0.8, description="Warn when quota usage exceeds this fraction")


class CacheConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable result caching")
    ttl_hours: int = Field(default=24, description="Cache TTL in hours")


class RepoContextConfig(BaseModel):
    enabled: bool = Field(default=True, description="Weave unchanged repo files into review context")
    max_anchors: int = Field(default=8, description="Max diff anchors to search (token + CPU budget)")
    max_scan_files: int = Field(default=120, description="Max source files to read per review")
    max_files_per_anchor: int = Field(default=3, description="Stop searching each anchor after N hits")
    max_snippets: int = Field(default=5, description="Max resonant evidence lines in prompt")
    max_deps: int = Field(default=12, description="Max declared dependencies in context block")
    max_chars: int = Field(default=2800, description="Character budget for repository context block")


class ClaimVerifierConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="Post-filter structural LLM findings against PR-head symbol maps",
    )
    include_symbol_index: bool = Field(
        default=True,
        description="Inject a compact post-change symbol index into the review prompt",
    )
    max_index_chars: int = Field(
        default=2400,
        description="Character budget for the symbol index prompt block",
    )


class FindingAdjudicatorConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="LLM post-pass to drop speculative nits; keep actionable scale gaps",
    )


class ProviderRuntimeConfig(BaseModel):
    rpm: float = Field(default=30.0, description="Requests per minute token-bucket rate")
    max_inflight: int = Field(default=1, description="Max concurrent calls to this provider")
    max_context_tokens: int = Field(default=8000, description="Max chunk tokens this provider accepts")
    quality_prior: float = Field(default=0.8, description="Cold-start quality prior for market score")
    nominal_latency_ms: float = Field(default=800.0, description="Expected latency for scoring")


class ProvidersConfig(BaseModel):
    groq: ProviderRuntimeConfig = Field(
        default_factory=lambda: ProviderRuntimeConfig(
            rpm=30, max_inflight=2, max_context_tokens=7000, quality_prior=0.9, nominal_latency_ms=400
        )
    )
    openrouter: ProviderRuntimeConfig = Field(
        default_factory=lambda: ProviderRuntimeConfig(
            rpm=20, max_inflight=2, max_context_tokens=100000, quality_prior=0.7, nominal_latency_ms=800
        )
    )
    gemini: ProviderRuntimeConfig = Field(
        default_factory=lambda: ProviderRuntimeConfig(
            rpm=10, max_inflight=1, max_context_tokens=200000, quality_prior=0.85, nominal_latency_ms=1200
        )
    )


class SchedulerConfig(BaseModel):
    wall_clock_sec: int = Field(default=720, description="Max seconds for scheduled review")
    max_workers: int = Field(default=4, description="Upper bound on adaptive concurrency")
    health_threshold: float = Field(default=25.0, description="Skip providers below this health score")
    partial_on_exhausted: bool = Field(
        default=True, description="Return partial review when providers are exhausted"
    )
    max_capacity_wait_sec: float = Field(
        default=120.0,
        description=(
            "Max seconds spent sleeping for provider cooldowns before early-deferring "
            "with NO_PROVIDER (still allows attempts/retries within this budget)"
        ),
    )
    auto_retry_delay_min_sec: int = Field(
        default=60, description="Min delay before one automatic /sorge retry after rate-limit"
    )
    auto_retry_delay_max_sec: int = Field(
        default=120, description="Max delay before one automatic /sorge retry after rate-limit"
    )
    gemini_dead_max_chunks: int = Field(
        default=8,
        description=(
            "When Gemini soft daily quota is exhausted, only review the top-N "
            "priority chunks on free-tier Groq/OpenRouter; defer the rest."
        ),
    )


class Config(BaseModel):
    sorge: dict[str, bool] = Field(default_factory=lambda: {"enabled": True})
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    groq: GroqConfig = Field(default_factory=GroqConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    quota: QuotaConfig = Field(default_factory=QuotaConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    repo_context: RepoContextConfig = Field(default_factory=RepoContextConfig)
    claim_verifier: ClaimVerifierConfig = Field(default_factory=ClaimVerifierConfig)
    finding_adjudicator: FindingAdjudicatorConfig = Field(
        default_factory=FindingAdjudicatorConfig
    )
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        data = toml.load(path)
        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        config = cls()

        if os.getenv("SORGE_ENABLED"):
            config.sorge["enabled"] = os.getenv("SORGE_ENABLED").lower() == "true"

        if os.getenv("SORGE_MIN_LINES"):
            config.filters.min_lines = int(os.getenv("SORGE_MIN_LINES"))

        if os.getenv("SORGE_GEMINI_ENABLED"):
            config.gemini.enabled = os.getenv("SORGE_GEMINI_ENABLED").lower() == "true"

        if os.getenv("SORGE_GEMINI_MODEL"):
            config.gemini.model = os.getenv("SORGE_GEMINI_MODEL")

        if os.getenv("SORGE_OPENROUTER_ENABLED"):
            config.openrouter.enabled = os.getenv("SORGE_OPENROUTER_ENABLED").lower() == "true"

        if os.getenv("SORGE_OPENROUTER_MODEL"):
            config.openrouter.model = os.getenv("SORGE_OPENROUTER_MODEL")

        if os.getenv("SORGE_OPENROUTER_ENDPOINT"):
            config.openrouter.endpoint = os.getenv("SORGE_OPENROUTER_ENDPOINT")

        if os.getenv("SORGE_OPENROUTER_API_KEY"):
            config.openrouter.api_key = os.getenv("SORGE_OPENROUTER_API_KEY")

        if os.getenv("SORGE_GROQ_ENABLED"):
            config.groq.enabled = os.getenv("SORGE_GROQ_ENABLED").lower() == "true"

        if os.getenv("SORGE_GROQ_MODEL"):
            config.groq.model = os.getenv("SORGE_GROQ_MODEL")

        if os.getenv("SORGE_GROQ_ENDPOINT"):
            config.groq.endpoint = os.getenv("SORGE_GROQ_ENDPOINT")

        if os.getenv("SORGE_GROQ_API_KEY"):
            config.groq.api_key = os.getenv("SORGE_GROQ_API_KEY")

        return config

    def to_dict(self) -> dict:
        return self.model_dump()