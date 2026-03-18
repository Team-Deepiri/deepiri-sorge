"""Tests for diff parser"""

import pytest

from bot.diff_parser import DiffParser, ParsedDiff, FileChange


@pytest.fixture
def parser():
    return DiffParser()


@pytest.fixture
def simple_diff():
    return """diff --git a/main.py b/main.py
index 1234567..abcdefg 100644
--- a/main.py
+++ b/main.py
@@ -1,5 +1,7 @@
 def hello():
-    print("hello")
+    print("hello world")
+    return True
     print("done")"""


@pytest.fixture
def multi_file_diff():
    return """diff --git a/main.py b/main.py
index 1234567..abcdefg 100644
--- a/main.py
+++ b/main.py
@@ -1,3 +1,4 @@
+import os
 def main():
     pass
diff --git a/utils.py b/utils.py
index 7654321..gfedcba 100644
--- a/utils.py
+++ b/utils.py
@@ -1,2 +1,3 @@
 def helper():
-    return None
+    return {"status": "ok"}
+"""


@pytest.fixture
def docs_diff():
    return """diff --git a/README.md b/README.md
index 1234567..abcdefg 100644
--- a/README.md
+++ b/README.md
@@ -1,2 +1,3 @@
 # My Project
+New line added"""


class TestDiffParser:
    def test_parse_empty_diff(self, parser):
        diff = ParsedDiff(
            raw="",
            files=[],
            file_changes={},
            lines_added=0,
            lines_deleted=0,
            files_changed=0,
        )
        
        result = parser.parse("")
        
        assert result.files == []
        assert result.lines_added == 0
        assert result.lines_deleted == 0
        assert result.files_changed == 0
    
    def test_parse_simple_diff(self, parser, simple_diff):
        result = parser.parse(simple_diff)
        
        assert "main.py" in result.files
        assert result.lines_added > 0
        assert result.lines_deleted > 0
    
    def test_parse_multi_file_diff(self, parser, multi_file_diff):
        result = parser.parse(multi_file_diff)
        
        assert result.files_changed == 2
        assert "main.py" in result.files
        assert "utils.py" in result.files
    
    def test_parse_detects_language(self, parser):
        diff = """diff --git a/script.py b/script.py
+print("hello")"""
        
        result = parser.parse(diff)
        
        assert "python" in result.language_counts
    
    def test_get_language_python(self, parser):
        lang = parser._detect_language("test.py")
        assert lang == "python"
    
    def test_get_language_javascript(self, parser):
        lang = parser._detect_language("test.js")
        assert lang == "javascript"
    
    def test_get_language_typescript(self, parser):
        lang = parser._detect_language("test.ts")
        assert lang == "typescript"
    
    def test_get_language_unknown(self, parser):
        lang = parser._detect_language("test.xyz")
        assert lang is None


class TestFileChange:
    def test_file_change_creation(self):
        change = FileChange(
            path="main.py",
            status="modified",
            additions=10,
            deletions=5,
        )
        
        assert change.path == "main.py"
        assert change.status == "modified"
        assert change.additions == 10
        assert change.deletions == 5


class TestParsedDiff:
    def test_get_summary(self):
        diff = ParsedDiff(
            raw="",
            files=["a.py", "b.py"],
            file_changes={},
            lines_added=100,
            lines_deleted=50,
            files_changed=2,
        )
        
        summary = diff.get_summary()
        
        assert "2 file(s)" in summary
        assert "100 additions" in summary
        assert "50 deletions" in summary
