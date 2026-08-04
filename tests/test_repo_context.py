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


def _write_cpp_repo(root: Path) -> None:
    """Mirrors deepiri-crankl: headers live under src/internal_headers/."""
    src = root / "src"
    (src / "internal_headers").mkdir(parents=True)
    (src / "internal_headers" / "mmap_region.hpp").write_text(
        """#pragma once
#include <sys/mman.h>

namespace crankl {

class MappedRegion {
 public:
  explicit MappedRegion(int fd);
  ~MappedRegion();
  void reset();

 private:
  void* base_;
  size_t size_;
};

}  // namespace crankl
"""
    )
    (src / "safetensors.cpp").write_text(
        """#include "internal_headers/mmap_region.hpp"

namespace crankl {
void load_shard() {}
}  // namespace crankl
"""
    )


def _cpp_config() -> RepoContextConfig:
    return RepoContextConfig()


def test_weaver_gathers_context_for_cpp_sources(tmp_path: Path):
    _write_cpp_repo(tmp_path)

    diff = DiffParser().parse(
        """diff --git a/src/mmap.cpp b/src/mmap.cpp
--- a/src/mmap.cpp
+++ b/src/mmap.cpp
@@ -7,0 +7,6 @@
+class MappedRegion {
+ public:
+  void reset();
+};
"""
    )

    pack = RepoContextWeaver(_cpp_config()).weave(tmp_path, diff)

    assert pack.anchors, "expected anchors from a C++ diff"
    assert "mappedregion" in pack.anchors
    assert any("mmap_region.hpp" in hit.path for hit in pack.hits)


def test_weaver_anchors_on_include_header_stem(tmp_path: Path):
    _write_cpp_repo(tmp_path)

    diff = DiffParser().parse(
        """diff --git a/src/loader.cpp b/src/loader.cpp
--- a/src/loader.cpp
+++ b/src/loader.cpp
@@ -1,0 +1,2 @@
+#include "internal_headers/mmap_region.hpp"
+void load();
"""
    )

    pack = RepoContextWeaver(_cpp_config()).weave(tmp_path, diff)

    # Header stem, not the ".hpp" suffix.
    assert "mmap_region" in pack.anchors
    assert "hpp" not in pack.anchors


def test_weaver_drops_cpp_keywords_as_anchors(tmp_path: Path):
    _write_cpp_repo(tmp_path)

    diff = DiffParser().parse(
        """diff --git a/src/mmap.cpp b/src/mmap.cpp
--- a/src/mmap.cpp
+++ b/src/mmap.cpp
@@ -1,0 +1,4 @@
+static constexpr size_t kPageSize = 4096;
+template <typename T>
+inline const T* as_const(T* p) noexcept { return p; }
"""
    )

    pack = RepoContextWeaver(_cpp_config()).weave(tmp_path, diff)

    for keyword in ("const", "constexpr", "static", "inline", "template", "typename", "noexcept"):
        assert keyword not in pack.anchors, f"{keyword} should be a stopword"
