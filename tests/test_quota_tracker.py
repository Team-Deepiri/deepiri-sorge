"""Tests for quota tracker."""

from bot.config import QuotaConfig
from bot.quota_tracker import QuotaTracker


def test_can_use_and_record():
    q = QuotaTracker(limits={"gemini": 2, "qwen": 10, "openrouter": 5}, used={"gemini": 0, "qwen": 0, "openrouter": 0})
    assert q.can_use("gemini")
    q.record("gemini")
    assert q.remaining("gemini") == 1
    q.record("gemini")
    assert not q.can_use("gemini")


def test_from_config():
    q = QuotaTracker.from_config(QuotaConfig())
    assert q.limits["gemini"] == 20
    assert q.can_use("openrouter")
