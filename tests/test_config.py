"""Tests for config management"""


from bot.config import Config, FiltersConfig, GeminiConfig, GroqConfig, OpenRouterConfig


class TestConfig:
    def test_default_config(self):
        config = Config()

        assert config.sorge.get("enabled", True) is True
        assert config.filters.min_lines == 20
        assert config.gemini.enabled is True
        assert config.gemini.model == "gemini-2.5-flash"
        assert config.openrouter.enabled is True
        assert config.openrouter.model == "google/gemma-4-31b-it:free"
        assert config.openrouter.endpoint == "https://openrouter.ai/api/v1/chat/completions"
        assert config.groq.enabled is True
        assert config.groq.model == "qwen/qwen3-32b"
        assert config.groq.endpoint == "https://api.groq.com/openai/v1/chat/completions"

    def test_filters_defaults(self):
        filters = FiltersConfig()

        assert filters.min_lines == 20
        assert filters.skip_docs is True
        assert filters.skip_deps is True
        assert filters.skip_tests is False

    def test_routing_defaults(self):
        config = Config()

        assert config.routing.small_pr_threshold == 3700
        assert config.routing.medium_pr_threshold == 50000
        assert config.routing.large_pr_threshold == 50000

    def test_openrouter_defaults(self):
        openrouter = OpenRouterConfig()

        assert openrouter.enabled is True
        assert openrouter.model == "google/gemma-4-31b-it:free"
        assert openrouter.endpoint == "https://openrouter.ai/api/v1/chat/completions"
        assert openrouter.api_key is None

    def test_groq_defaults(self):
        groq = GroqConfig()

        assert groq.enabled is True
        assert groq.model == "qwen/qwen3-32b"
        assert groq.endpoint == "https://api.groq.com/openai/v1/chat/completions"
        assert groq.api_key is None

    def test_gemini_defaults(self):
        gemini = GeminiConfig()

        assert gemini.enabled is True
        assert gemini.model == "gemini-2.5-flash"
        assert gemini.api_key is None

    def test_config_to_dict(self):
        config = Config()

        data = config.to_dict()

        assert "filters" in data
        assert "review" in data
        assert "gemini" in data
        assert "openrouter" in data
        assert "groq" in data


class TestConfigOverrides:
    def test_custom_min_lines(self):
        config = Config(filters=FiltersConfig(min_lines=50))

        assert config.filters.min_lines == 50

    def test_custom_gemini_model(self):
        config = Config(gemini=GeminiConfig(model="gemini-1.5-pro"))

        assert config.gemini.model == "gemini-1.5-pro"

    def test_custom_gemini_enabled(self):
        config = Config(gemini=GeminiConfig(enabled=False))

        assert config.gemini.enabled is False

    def test_custom_openrouter_model(self):
        config = Config(openrouter=OpenRouterConfig(model="meta-llama/llama-3-70b-instruct"))

        assert config.openrouter.model == "meta-llama/llama-3-70b-instruct"
