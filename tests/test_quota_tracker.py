"""Tests for quota tracker + daily file continuity."""

from pathlib import Path

from bot.config import QuotaConfig
from bot.quota_tracker import QuotaTracker


def test_can_use_and_record(tmp_path: Path):
    path = tmp_path / "quota.json"
    q = QuotaTracker(
        limits={"gemini": 2, "gpt": 10, "openrouter": 5},
        used={"gemini": 0, "gpt": 0, "openrouter": 0},
        persist_path=path,
        sync_remote=False,
    )
    assert q.can_use("gemini")
    q.record("gemini")
    assert q.remaining("gemini") == 1
    # Soft reserve blocks the last Gemini slot by default.
    assert not q.can_use("gemini")
    assert q.can_use("gemini", respect_soft_reserve=False)
    assert path.exists()
    q.record("gemini")
    assert not q.can_use("gemini")
    assert not q.can_use("gemini", respect_soft_reserve=False)


def test_soft_reserve_last_resort(tmp_path: Path):
    q = QuotaTracker(
        limits={"gemini": 20, "gpt": 10, "openrouter": 50},
        used={"gemini": 19, "gpt": 0, "openrouter": 0},
        persist_path=tmp_path / "quota.json",
        sync_remote=False,
    )
    assert q.remaining("gemini") == 1
    assert not q.can_use("gemini")
    assert q.can_use("gemini", respect_soft_reserve=False)


def test_from_config(tmp_path: Path):
    q = QuotaTracker.from_config(
        QuotaConfig(),
        persist_path=tmp_path / "q.json",
        sync_remote=False,
    )
    assert q.limits["gemini"] == 20
    assert q.can_use("openrouter")


def test_quota_file_seeds_next_tracker(tmp_path: Path, monkeypatch):
    path = tmp_path / "quota_daily.json"
    monkeypatch.delenv("SORGE_GEMINI_USED_TODAY", raising=False)
    q1 = QuotaTracker.from_config(
        QuotaConfig(), persist_path=path, sync_remote=False
    )
    q1.record("gemini")
    q1.record("gemini")
    q2 = QuotaTracker.from_config(
        QuotaConfig(), persist_path=path, sync_remote=False
    )
    assert q2.used["gemini"] >= 2
    assert q2.remaining("gemini") == 20 - q2.used["gemini"]
