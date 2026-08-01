"""Tests for file splitter."""

from bot.diff_parser import DiffParser, estimate_tokens
from bot.file_splitter import FileSplitter


def test_small_diff_single_chunk():
    raw = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
    diff = DiffParser().parse(raw)
    chunks = FileSplitter(chunk_budget=180_000, max_chunk_tokens=200_000).split(diff)
    assert len(chunks) == 1
    assert chunks[0].files == ["a.py"]


def test_estimate_tokens():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("x" * 400) == 100


def test_slice_files():
    raw = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-a\n+b\n"
    )
    diff = DiffParser().parse(raw)
    sub = diff.slice_files(["b.py"])
    assert sub.files == ["b.py"]
    assert "b.py" in sub.raw
    assert "a.py" not in sub.files


def test_oversized_pr_splits_multiple_chunks():
    part_a = "diff --git a/big.py b/big.py\n--- a/big.py\n+++ b/big.py\n" + "+line\n" * 8000
    part_b = "diff --git a/other.py b/other.py\n--- a/other.py\n+++ b/other.py\n" + "+x\n" * 8000
    diff = DiffParser().parse(part_a + part_b)
    splitter = FileSplitter(chunk_budget=5000, max_chunk_tokens=200_000)
    chunks = splitter.split(diff)
    assert len(chunks) >= 2


def test_many_unrelated_small_files_get_bin_packed():
    """A monorepo PR touching lots of unrelated files (no detected import
    edges between them — CSS, JSON, small components in different packages)
    used to become one single-file chunk per file. They should now get
    packed together up to chunk_budget instead of one request each."""
    parts = []
    for i in range(40):
        parts.append(
            f"diff --git a/pkg{i}/file{i}.css b/pkg{i}/file{i}.css\n"
            f"--- a/pkg{i}/file{i}.css\n+++ b/pkg{i}/file{i}.css\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        )
    raw = "".join(parts)
    diff = DiffParser().parse(raw)
    # Force the "too big for one chunk" path so dependency grouping runs,
    # but keep chunk_budget generous enough that plenty of these 40 tiny
    # files fit together in a single chunk.
    splitter = FileSplitter(chunk_budget=200, max_chunk_tokens=200_000)
    assert estimate_tokens(raw) > splitter.chunk_budget  # confirms split() path is exercised
    chunks = splitter.split(diff)
    # 40 unrelated files must not become 40 single-file chunks.
    assert len(chunks) < 40
    assert all(len(c.files) >= 1 for c in chunks)
    # No file dropped or duplicated across chunks.
    all_files = [f for c in chunks for f in c.files]
    assert sorted(all_files) == sorted(diff.files)
    assert len(all_files) == len(set(all_files))
