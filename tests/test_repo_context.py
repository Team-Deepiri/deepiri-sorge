"""Tests for diff-anchored repository context weaving."""

import os
from pathlib import Path

import pytest

from bot.config import RepoContextConfig
from bot.diff_parser import DiffParser
from bot.repo_context import RepoContextWeaver


def _write_repo(root: Path) -> None:
    (root / "requirements.txt").write_text("requests>=2.31\nloguru>=0.7\n")
    utils = root / "bot" / "utils"
    utils.mkdir(parents=True)
    (utils / "cache.py").write_text(
        '''"""Disk-backed cache for review results."""

def get(diff_raw: str, model: str, ttl_hours: int = 24) -> dict | None:
    return None

def set(diff_raw: str, model: str, result: dict) -> None:
    pass
'''
    )


def test_weaver_finds_existing_utility_for_duplicate_logic(tmp_path: Path):
    _write_repo(tmp_path)

    diff = DiffParser().parse(
        """diff --git a/bot/feature.py b/bot/feature.py
--- a/bot/feature.py
+++ b/bot/feature.py
@@ -1,0 +1,6 @@
+def get_cached_review(diff_raw, model):
+    store = {}
+    return store.get(diff_raw)
"""
    )

    pack = RepoContextWeaver(
        RepoContextConfig(enabled=True, max_snippets=5, max_scan_files=50)
    ).weave(tmp_path, diff)

    hit_paths = {h.path for h in pack.hits}
    assert any(os.path.normpath("bot/utils/cache.py") == os.path.normpath(p) for p in hit_paths)
    assert len(pack.text) < 2800
    assert pack.text.startswith("REPO_CONTEXT")
    assert "reuse|" in pack.text
    assert pack.fingerprint


def test_weaver_disabled_returns_empty_pack(tmp_path: Path):
    _write_repo(tmp_path)
    diff = DiffParser().parse("diff --git a/a.py b/a.py\n+print('x')")

    pack = RepoContextWeaver(RepoContextConfig(enabled=False)).weave(tmp_path, diff)

    assert pack.text == ""
    assert pack.hits == []


def test_char_budget_limits_output(tmp_path: Path):
    _write_repo(tmp_path)
    for i in range(20):
        (tmp_path / "bot" / "utils" / f"helper_{i}.py").write_text(
            f"def get_cached_review_{i}():\n    return {i}\n"
        )

    diff = DiffParser().parse(
        "diff --git a/bot/x.py b/bot/x.py\n+def get_cached_review():\n    pass\n"
    )
    pack = RepoContextWeaver(
        RepoContextConfig(enabled=True, max_chars=400, max_snippets=20)
    ).weave(tmp_path, diff)

    assert len(pack.text) <= 400


def test_build_prompt_includes_repo_context():
    from bot.diff_parser import DiffParser
    from bot.runners.base import BaseRunner

    class StubRunner(BaseRunner):
        model = "stub"

        def _run_review(self, diff):
            return None

    runner = StubRunner()
    runner._repo_context = "REPO_CONTEXT\nreuse|bot/utils/cache.py:5|def get(...)"
    prompt = runner._build_prompt(
        DiffParser().parse("diff --git a/a.py b/a.py\n+def foo():\n    pass")
    )

    assert "REPOSITORY CONTEXT" in prompt
    assert "bot/utils/cache.py" in prompt
    assert "System-fit rules" in prompt
