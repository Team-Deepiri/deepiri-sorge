"""Tests for symbol index + conservative claim verifier."""

import subprocess
from pathlib import Path

from bot.claim_verifier import ClaimVerifier
from bot.schemas import ReviewIssue, ReviewResult
from bot.symbol_index import SymbolIndexer, format_symbol_index


def _write_compiler(root: Path) -> Path:
    path = root / "quantum_core" / "compiler" / "compiler.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        '''"""QUASAR compiler with backward-compatible alias."""


class QUASAR:
    """Primary compiler."""

    def compile(self):
        return []


QuantumCompiler = QUASAR
'''
    )
    return path


def test_symbol_indexer_finds_class_and_alias(tmp_path: Path):
    _write_compiler(tmp_path)
    indexes = SymbolIndexer().index_files(
        tmp_path, ["quantum_core/compiler/compiler.py"]
    )
    assert len(indexes) == 1
    names = indexes[0].names()
    assert "QUASAR" in names
    assert "QuantumCompiler" in names
    alias = indexes[0].first_binding("QuantumCompiler")
    assert alias is not None
    assert alias.kind == "alias"
    assert alias.target == "QUASAR"
    quasar = indexes[0].first_binding("QUASAR")
    assert quasar is not None
    assert quasar.line < alias.line


def test_format_symbol_index_includes_aliases(tmp_path: Path):
    _write_compiler(tmp_path)
    indexes = SymbolIndexer().index_files(
        tmp_path, ["quantum_core/compiler/compiler.py"]
    )
    text = format_symbol_index(indexes)
    assert "SYMBOL_INDEX" in text
    assert "QuantumCompiler -> QUASAR" in text
    assert "QUASAR @" in text


def test_suppresses_nameerror_when_alias_bound_before_cited_line(tmp_path: Path):
    _write_compiler(tmp_path)
    # Alias is at the last line; QUASAR is defined earlier — classic false positive.
    alias_line = (
        tmp_path / "quantum_core" / "compiler" / "compiler.py"
    ).read_text().splitlines()
    cited = next(
        i for i, line in enumerate(alias_line, start=1) if "QuantumCompiler" in line
    )

    result = ReviewResult(
        summary="rename",
        issues=[
            ReviewIssue(
                severity="high",
                file="quantum_core/compiler/compiler.py",
                line=cited,
                message=(
                    "Alias logic reassigns QUASAR = QuantumCompiler before "
                    "QuantumCompiler is defined, raising a NameError on import."
                ),
                suggestion="Use QuantumCompiler = QUASAR instead.",
            ),
            ReviewIssue(
                severity="low",
                file="quantum_core/compiler/__init__.py",
                line=121,
                message="`__all__` list contains duplicate 'QUASAR' entry.",
            ),
        ],
        recommendations=[],
        score=7.8,
        latency_ms=1.0,
        model="test",
    )

    verified = ClaimVerifier().verify_result(
        result,
        repo_root=tmp_path,
        changed_paths=["quantum_core/compiler/compiler.py"],
    )
    assert len(verified.issues) == 1
    assert "duplicate" in verified.issues[0].message.lower()
    assert verified.routing_meta["claim_verifier"]["suppressed"] == 1


def test_keeps_finding_when_symbol_truly_missing(tmp_path: Path):
    src = tmp_path / "pkg" / "mod.py"
    src.parent.mkdir(parents=True)
    src.write_text("def hello():\n    return MissingHelper()\n")

    result = ReviewResult(
        summary="bug",
        issues=[
            ReviewIssue(
                severity="high",
                file="pkg/mod.py",
                line=2,
                message="NameError: `MissingHelper` is undefined.",
            )
        ],
        recommendations=[],
        score=5.0,
        latency_ms=1.0,
        model="test",
    )
    verified = ClaimVerifier().verify_result(result, repo_root=tmp_path)
    assert len(verified.issues) == 1
    assert verified.routing_meta is None or not verified.routing_meta.get(
        "claim_verifier", {}
    ).get("suppressed")


def test_keeps_finding_when_uncertain_conditional_not_analyzed(tmp_path: Path):
    """Verifier must not invent control-flow analysis — uncertain → keep."""
    src = tmp_path / "a.py"
    src.write_text(
        "if False:\n"
        "    Helper = int\n"
        "\n"
        "x = Helper\n"
    )
    result = ReviewResult(
        summary="maybe",
        issues=[
            ReviewIssue(
                severity="high",
                file="a.py",
                line=4,
                message="NameError: `Helper` may be undefined at runtime.",
            )
        ],
        recommendations=[],
        score=6.0,
        latency_ms=1.0,
        model="test",
    )
    # Module-level `Helper = int` is inside `if` — AST only walks top-level body,
    # so Helper is NOT in the index → uncertain → keep.
    verified = ClaimVerifier().verify_result(result, repo_root=tmp_path)
    assert len(verified.issues) == 1


def test_confidence_rule_requires_binding_before_cited_line(tmp_path: Path):
    src = tmp_path / "order.py"
    src.write_text(
        "def use():\n"
        "    return Later()\n"
        "\n"
        "class Later:\n"
        "    pass\n"
    )
    # Class is at line 4; use cites line 2 — binding is AFTER cited line → keep.
    result = ReviewResult(
        summary="order",
        issues=[
            ReviewIssue(
                severity="high",
                file="order.py",
                line=2,
                message="`Later` is used before it is defined (NameError risk).",
            )
        ],
        recommendations=[],
        score=6.0,
        latency_ms=1.0,
        model="test",
    )
    verified = ClaimVerifier().verify_result(result, repo_root=tmp_path)
    assert len(verified.issues) == 1


def test_suppresses_missing_npm_deps_when_declared_on_head(tmp_path: Path):
    (tmp_path / "package.json").write_text(
        '{"dependencies": {"mammoth": "^1.6.0", "pdf-parse": "^1.1.1"}}'
    )
    result = ReviewResult(
        summary="deps",
        issues=[
            ReviewIssue(
                severity="high",
                file="src/services/documentService.ts",
                line=204,
                message=(
                    "The documentService.ts file now uses `pdf-parse` and `mammoth` "
                    "for local text extraction. However, these new dependencies are "
                    "not listed in `package.json` or `package-lock.json` within this "
                    "diff. This will lead to runtime errors if not explicitly added."
                ),
                suggestion=(
                    "Add `pdf-parse` and `mammoth` to package.json and run npm install."
                ),
            ),
            ReviewIssue(
                severity="medium",
                file="src/routes/documentRoutes.ts",
                line=109,
                message="Simplistic documentType defaulting to PDF for unknown MIME types.",
            ),
        ],
        recommendations=[],
        score=7.8,
        latency_ms=1.0,
        model="test",
    )
    verified = ClaimVerifier().verify_result(result, repo_root=tmp_path)
    assert len(verified.issues) == 1
    assert verified.issues[0].file == "src/routes/documentRoutes.ts"
    assert verified.routing_meta["claim_verifier"]["suppressed"] == 1


def test_keeps_missing_dep_claim_when_package_absent(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"dependencies": {"express": "^4.0.0"}}')
    result = ReviewResult(
        summary="deps",
        issues=[
            ReviewIssue(
                severity="high",
                file="src/a.ts",
                line=1,
                message=(
                    "`left-pad` is not listed in package.json and will lead to "
                    "runtime errors."
                ),
            )
        ],
        recommendations=[],
        score=6.0,
        latency_ms=1.0,
        model="test",
    )
    verified = ClaimVerifier().verify_result(result, repo_root=tmp_path)
    assert len(verified.issues) == 1


def _write_mmap_cpp(root: Path) -> Path:
    """Mirrors deepiri-crankl mmap.cpp: includes sit above the first diff hunk."""
    path = root / "src" / "mmap.cpp"
    path.parent.mkdir(parents=True)
    path.write_text(
        """#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>

namespace crankl {

MappedRegion open_region(const char* path) {
    int fd = open(path, O_RDONLY);
    return MappedRegion(fd);
}

}  // namespace crankl
"""
    )
    return path


def test_include_claim_suppressed_when_header_already_present(tmp_path: Path):
    _write_mmap_cpp(tmp_path)
    issue = ReviewIssue(
        severity="high",
        file="src/mmap.cpp",
        line=9,
        message="`open()` is used but `#include <fcntl.h>` is missing.",
        suggestion="Add `#include <fcntl.h>` at the top of the file.",
    )
    report = ClaimVerifier().verify_issues([issue], repo_root=tmp_path)
    assert report.kept == []
    assert report.suppressed_count == 1
    assert "fcntl.h" in report.suppressed[0].reason


def test_include_claim_matches_on_bare_filename(tmp_path: Path):
    _write_mmap_cpp(tmp_path)
    issue = ReviewIssue(
        severity="medium",
        file="src/mmap.cpp",
        line=9,
        message="Missing header: mman.h is required for munmap().",
    )
    report = ClaimVerifier().verify_issues([issue], repo_root=tmp_path)
    assert report.kept == []
    assert report.suppressed_count == 1


def test_include_claim_kept_when_header_genuinely_absent(tmp_path: Path):
    _write_mmap_cpp(tmp_path)
    issue = ReviewIssue(
        severity="high",
        file="src/mmap.cpp",
        line=9,
        message="`std::vector` is used but `#include <vector>` is missing.",
    )
    report = ClaimVerifier().verify_issues([issue], repo_root=tmp_path)
    assert report.kept == [issue]
    assert report.suppressed_count == 0


def test_include_claim_kept_when_only_some_headers_present(tmp_path: Path):
    _write_mmap_cpp(tmp_path)
    issue = ReviewIssue(
        severity="high",
        file="src/mmap.cpp",
        line=9,
        message="Missing `#include <fcntl.h>` and `#include <vector>`.",
    )
    report = ClaimVerifier().verify_issues([issue], repo_root=tmp_path)
    assert report.kept == [issue]


def test_include_claim_ignored_for_non_cpp_files(tmp_path: Path):
    _write_compiler(tmp_path)
    issue = ReviewIssue(
        severity="low",
        file="quantum_core/compiler/compiler.py",
        line=3,
        message="missing import header for QUASAR",
    )
    report = ClaimVerifier().verify_issues([issue], repo_root=tmp_path)
    assert report.kept == [issue]


def test_include_claim_uncertain_when_file_missing_on_head(tmp_path: Path):
    issue = ReviewIssue(
        severity="high",
        file="src/gone.cpp",
        line=4,
        message="`open()` used without `#include <fcntl.h>`.",
    )
    report = ClaimVerifier().verify_issues([issue], repo_root=tmp_path)
    assert report.kept == [issue]


_MMAP_HPP = """#pragma once
#include <sys/mman.h>
#include <cstddef>

#define CRANKL_PAGE_SIZE 4096

namespace crankl {

using RegionSize = std::size_t;

class MappedRegion {
 public:
  explicit MappedRegion(int fd);
  ~MappedRegion() { reset(); }
  void reset();

 private:
  void* base_;
  RegionSize size_;
};

struct ShardHeader {
  RegionSize offset;
};

}  // namespace crankl
"""


def _write_mmap_header(root: Path) -> Path:
    path = root / "src" / "internal_headers" / "mmap_region.hpp"
    path.parent.mkdir(parents=True)
    path.write_text(_MMAP_HPP)
    return path


def test_symbol_indexer_parses_cpp_header(tmp_path: Path):
    _write_mmap_header(tmp_path)
    indexes = SymbolIndexer().index_files(
        tmp_path, ["src/internal_headers/mmap_region.hpp"]
    )
    assert len(indexes) == 1
    names = indexes[0].names()

    assert "MappedRegion" in names
    assert "ShardHeader" in names
    assert "reset" in names
    assert "~MappedRegion" in names
    assert "RegionSize" in names
    assert "CRANKL_PAGE_SIZE" in names
    # #include anchors on the header stem.
    assert "mman" in names

    region = indexes[0].first_binding("MappedRegion")
    assert region is not None and region.kind == "class"
    alias = indexes[0].first_binding("RegionSize")
    assert alias is not None and alias.kind == "alias"


def test_symbol_indexer_skips_cpp_control_flow_and_comments(tmp_path: Path):
    path = tmp_path / "src" / "loader.cpp"
    path.parent.mkdir(parents=True)
    path.write_text(
        """#include "mmap_region.hpp"

/* class CommentedOut {
   void ghost();
 }; */

void load_shard(int fd) {
  if (fd < 0) { return; }
  for (int i = 0; i < 4; ++i) {
    while (i > 2) { break; }
  }
  // void commented_fn();
}
"""
    )
    names = SymbolIndexer().index_files(tmp_path, ["src/loader.cpp"])[0].names()

    assert "load_shard" in names
    for absent in ("if", "for", "while", "return", "ghost", "commented_fn", "CommentedOut"):
        assert absent not in names


def test_verifier_disproves_undefined_symbol_claim_in_cpp(tmp_path: Path):
    _write_mmap_header(tmp_path)
    issue = ReviewIssue(
        severity="high",
        file="src/internal_headers/mmap_region.hpp",
        line=20,
        message="`RegionSize` is not defined anywhere in this file.",
    )
    report = ClaimVerifier().verify_issues([issue], repo_root=tmp_path)
    assert report.kept == []
    assert report.suppressed_count == 1


def test_verifier_keeps_undefined_claim_for_absent_cpp_symbol(tmp_path: Path):
    _write_mmap_header(tmp_path)
    issue = ReviewIssue(
        severity="high",
        file="src/internal_headers/mmap_region.hpp",
        line=20,
        message="`ShardFooter` is not defined anywhere in this file.",
    )
    report = ClaimVerifier().verify_issues([issue], repo_root=tmp_path)
    assert report.kept == [issue]


def _write_cmake_tree(root: Path) -> None:
    """Mirrors deepiri-crankl: internal_headers lives under src/, not at root."""
    (root / "src" / "internal_headers").mkdir(parents=True)
    (root / "src" / "internal_headers" / "simd.hpp").write_text("#pragma once\n")
    (root / "tests" / "ctest").mkdir(parents=True)
    (root / "tests" / "ctest" / "test_simd.cpp").write_text(
        '#include "internal_headers/simd.hpp"\nint main() { return 0; }\n'
    )
    (root / "tests" / "CMakeLists.txt").write_text(
        "target_include_directories(test_simd PRIVATE ${CMAKE_SOURCE_DIR}/src)\n"
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)


def test_path_claim_suppressed_when_directory_exists_under_another_parent(tmp_path: Path):
    """deepiri-crankl#28: suggested ${CMAKE_SOURCE_DIR}/internal_headers, but the
    directory is really src/internal_headers and already on the include path."""
    _write_cmake_tree(tmp_path)
    issue = ReviewIssue(
        severity="critical",
        file="tests/ctest/test_simd.cpp",
        line=2,
        message=(
            'The test includes "internal_headers/simd.hpp" but the CMakeLists.txt '
            "only adds ${CMAKE_SOURCE_DIR}/src to the include directories, so the "
            "internal header cannot be found."
        ),
        suggestion=(
            "Add `target_include_directories(test_simd PRIVATE "
            "${CMAKE_SOURCE_DIR}/internal_headers)` in tests/CMakeLists.txt."
        ),
    )
    report = ClaimVerifier().verify_issues([issue], repo_root=tmp_path)
    assert report.kept == []
    assert report.suppressed_count == 1
    assert "src/internal_headers" in report.suppressed[0].reason


def test_path_claim_kept_when_directory_genuinely_absent(tmp_path: Path):
    _write_cmake_tree(tmp_path)
    issue = ReviewIssue(
        severity="high",
        file="tests/ctest/test_simd.cpp",
        line=2,
        message=(
            "The include directory ${CMAKE_SOURCE_DIR}/third_party/eigen does not "
            "exist, so the build cannot be configured."
        ),
    )
    report = ClaimVerifier().verify_issues([issue], repo_root=tmp_path)
    assert report.kept == [issue]
    assert report.suppressed_count == 0


def test_path_claim_ignored_without_a_missing_assertion(tmp_path: Path):
    """A finding that merely mentions include directories is not a path claim."""
    _write_cmake_tree(tmp_path)
    issue = ReviewIssue(
        severity="low",
        file="tests/CMakeLists.txt",
        line=1,
        message="Consider grouping the include directories for readability.",
    )
    report = ClaimVerifier().verify_issues([issue], repo_root=tmp_path)
    assert report.kept == [issue]
