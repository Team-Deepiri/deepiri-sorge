"""Tests for compact IMPORT_VS_MANIFEST evidence."""

from pathlib import Path

from bot.diff_parser import DiffParser
from bot.manifest_evidence import (
    extract_imported_packages,
    format_import_manifest_evidence,
    load_declared_dependencies,
)


def test_extract_js_imports_skips_relative():
    raw = """diff --git a/src/a.ts b/src/a.ts
--- a/src/a.ts
+++ b/src/a.ts
@@ -0,0 +1,4 @@
+import mammoth from "mammoth";
+import { parse } from "pdf-parse";
+import local from "./local";
+import express from "express";
"""
    diff = DiffParser().parse(raw)
    pkgs = extract_imported_packages(diff)
    assert pkgs == ["mammoth", "pdf-parse", "express"]


def test_format_import_manifest_only_lists_diff_imports(tmp_path: Path):
    (tmp_path / "package.json").write_text(
        '{"dependencies": {"mammoth": "1", "pdf-parse": "1", "left-pad": "1", '
        '"express": "4", "lodash": "4", "webpack": "5", "react": "18"}}'
    )
    raw = """diff --git a/src/a.ts b/src/a.ts
--- a/src/a.ts
+++ b/src/a.ts
@@ -0,0 +1,2 @@
+import mammoth from "mammoth";
+import ghost from "not-installed-pkg";
"""
    diff = DiffParser().parse(raw)
    block = format_import_manifest_evidence(tmp_path, diff)
    assert "IMPORT_VS_MANIFEST" in block
    assert "mammoth: declared" in block
    assert "not-installed-pkg: NOT declared" in block
    # Must not dump unrelated declared packages into the prompt.
    assert "lodash" not in block
    assert "webpack" not in block
    assert "react" not in block


def test_load_declared_dependencies_reads_package_json(tmp_path: Path):
    (tmp_path / "package.json").write_text(
        '{"dependencies": {"mammoth": "^1.6.0"}, "devDependencies": {"vitest": "^1"}}'
    )
    names = load_declared_dependencies(tmp_path)
    assert "mammoth" in names
    assert "vitest" in names
