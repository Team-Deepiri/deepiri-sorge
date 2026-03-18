"""Tests for config management"""

import pytest
from pathlib import Path

from bot.config import Config, FiltersConfig, GPUConfig, ModelConfig


class TestConfig:
    def test_default_config(self):
        config = Config()
        
        assert config.sorge.get("enabled", True) is True
        assert config.filters.min_lines == 20
        assert config.filters.max_cpu_lines == 500
        assert config.gpu.enabled is False
    
    def test_filters_defaults(self):
        filters = FiltersConfig()
        
        assert filters.min_lines == 20
        assert filters.skip_docs is True
        assert filters.skip_deps is True
        assert filters.skip_tests is False
    
    def test_gpu_defaults(self):
        gpu = GPUConfig()
        
        assert gpu.enabled is False
        assert gpu.threshold_lines == 1000
        assert gpu.endpoint == ""
        assert gpu.timeout == 60
    
    def test_model_defaults(self):
        model = ModelConfig()
        
        assert model.name == "llama-7b-q4"
        assert model.context_size == 4096
        assert model.threads == 4
    
    def test_config_to_dict(self):
        config = Config()
        
        data = config.to_dict()
        
        assert "filters" in data
        assert "gpu" in data
        assert "model" in data
        assert "review" in data


class TestConfigOverrides:
    def test_custom_min_lines(self):
        config = Config(filters=FiltersConfig(min_lines=50))
        
        assert config.filters.min_lines == 50
    
    def test_custom_gpu_enabled(self):
        config = Config(gpu=GPUConfig(enabled=True))
        
        assert config.gpu.enabled is True
    
    def test_custom_model_name(self):
        config = Config(model=ModelConfig(name="codellama-13b"))
        
        assert config.model.name == "codellama-13b"
