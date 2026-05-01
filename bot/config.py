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
    max_cpu_lines: int = Field(default=500, description="Max lines for CPU review")


class ReviewConfig(BaseModel):
    style: str = Field(default="concise", description="Review style: concise, detailed, minimal")
    languages: list[str] = Field(default_factory=lambda: ["*"], description="Languages to review")
    include_security: bool = Field(default=True, description="Include security checks")
    include_performance: bool = Field(default=True, description="Include performance checks")
    include_style: bool = Field(default=True, description="Include style checks")


class GPUConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable GPU fallback")
    threshold_lines: int = Field(default=1000, description="Line threshold for GPU")
    endpoint: str = Field(default="", description="GPU endpoint URL")
    api_key: str | None = Field(default=None, description="GPU API key")
    timeout: int = Field(default=60, description="Timeout in seconds")


class ModelConfig(BaseModel):
    name: str = Field(default="llama-7b-q4", description="Model name")
    path: str | None = Field(default=None, description="Path to model files")
    context_size: int = Field(default=4096, description="Model context size")
    threads: int = Field(default=4, description="CPU threads for inference")


class GitHubModelsConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable GitHub Models")
    model: str = Field(default="gpt-4o", description="Model to use")
    api_key: str | None = Field(default=None, description="API key (uses GITHUB_TOKEN env if not set)")


class GeminiConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable Gemini")
    model: str = Field(default="gemini-2.5-flash", description="Model to use")
    api_key: str | None = Field(default=None, description="API key (uses GOOGLE_API_KEY env if not set)")


class RoutingConfig(BaseModel):
    small_pr_threshold: int = Field(default=10000, description="Max tokens for small PR (GitHub Models)")
    large_pr_threshold: int = Field(default=25000, description="Min tokens for large PR (Gemini)")


class CacheConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable result caching")
    ttl_hours: int = Field(default=24, description="Cache TTL in hours")


class Config(BaseModel):
    sorge: dict[str, bool] = Field(default_factory=lambda: {"enabled": True})
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    github_models: GitHubModelsConfig = Field(default_factory=GitHubModelsConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

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

        if os.getenv("SORGE_MAX_CPU_LINES"):
            config.filters.max_cpu_lines = int(os.getenv("SORGE_MAX_CPU_LINES"))

        if os.getenv("SORGE_GPU_ENABLED"):
            config.gpu.enabled = os.getenv("SORGE_GPU_ENABLED").lower() == "true"

        if os.getenv("SORGE_GPU_ENDPOINT"):
            config.gpu.endpoint = os.getenv("SORGE_GPU_ENDPOINT")

        if os.getenv("SORGE_GPU_API_KEY"):
            config.gpu.api_key = os.getenv("SORGE_GPU_API_KEY")

        if os.getenv("SORGE_MODEL_PATH"):
            config.model.path = os.getenv("SORGE_MODEL_PATH")

        if os.getenv("SORGE_GITHUB_MODELS_ENABLED"):
            config.github_models.enabled = os.getenv("SORGE_GITHUB_MODELS_ENABLED").lower() == "true"

        if os.getenv("SORGE_GITHUB_MODELS_MODEL"):
            config.github_models.model = os.getenv("SORGE_GITHUB_MODELS_MODEL")

        if os.getenv("SORGE_GEMINI_ENABLED"):
            config.gemini.enabled = os.getenv("SORGE_GEMINI_ENABLED").lower() == "true"

        if os.getenv("SORGE_GEMINI_MODEL"):
            config.gemini.model = os.getenv("SORGE_GEMINI_MODEL")

        return config

    def to_dict(self) -> dict:
        return self.model_dump()
