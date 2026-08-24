"""Conservative post-LLM claim verifier for structural / dependency false positives.

Policy (confidence rule):
  Suppress ONLY when the verifier can positively establish that the claim is
  false against the PR-head checkout (``confidence == "disproven"``). Otherwise
  preserve the original finding (``uncertain`` / ``confirmed``).

  1. Structural / NameError claims → symbol index (bound before cited line).
  2. Missing-dependency claims → package.json / requirements / pyproject on HEAD.
  3. Missing-``#include`` claims → the C/C++ file's own include block on HEAD.
  4. "Path X does not exist" claims → the tracked file/directory list on HEAD.
  5. Missing-import claims, any language → the file's own import lines on HEAD.
  6. Predicted build/compile failures → CI's verdict on the PR head.

Rationale for (3) and (5): a diff only shows changed hunks plus a few lines of
context. An import above the first hunk is invisible to the model, so "you use
open() without including <fcntl.h>" is a reasonable inference from an
incomplete picture. Checking the file on disk costs zero prompt tokens, where
widening the diff window would cost them on every chunk.

(5) generalises (3) rather than replacing it. The C-specific path understands
header basenames (``<sys/mman.h>`` cited as ``mman.h``); the generic path knows
only that imports sit on their own line and start with one of a small set of
keywords, which is enough for Rust/Go/Java/C#/JS without a parser per language.

Rationale for (6): predicting a compile error is a claim the real compiler has
already settled. Only a definite green build disproves it, and a green build is
evidence about compilation *only* — never about leaks, races, or logic.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from loguru import logger

from bot.manifest_evidence import load_declared_dependencies
from bot.schemas import ReviewIssue, ReviewResult, compute_score_from_issues
from bot.symbol_index import FileSymbolIndex, SymbolIndexer

# Findings that look like structural / NameError / undefined-symbol claims.
_STRUCTURAL_RE = re.compile(
    r"(?i)\b("
    r"name\s*error|nameerror|"
    r"undefined|not\s+defined|before\s+(?:it\s+is\s+)?defined|"
    r"used\s+before\s+(?:assignment|definition|defined)|"
    r"forward\s+reference|does\s+not\s+exist|missing\s+(?:name|symbol|import)|"
    r"cannot\s+be\s+resolved|unresolved\s+reference|"
    r"raises?\s+at\s+import"
    r")\b"
)

# "X is not in package.json / requirements / lockfile" style claims.
#
# Models often describe the manifest generically — "the project's dependency
# manifests", "the dependency list" — without naming a file. Requiring a
# literal filename here let three false positives through on a HELOX PR where
# transformers/numpy/pytest were all declared in pyproject.toml. Match the
# concept, not the spelling.
_DEPENDENCY_MANIFEST_RE = re.compile(
    r"(?i)\b("
    r"package\.json|package-lock\.json|npm|yarn\.lock|pnpm-lock|"
    r"requirements(?:\.txt)?|pyproject\.toml|poetry\.lock|Pipfile(?:\.lock)?|"
    r"dependency\s+manifests?|dependenc(?:y|ies)\s+list|manifest\s+files?|"
    r"dependency\s+(?:declarations?|specifications?)|"
    r"declared\s+dependenc(?:y|ies)|project'?s?\s+dependenc(?:y|ies)"
    r")\b"
)
_DEPENDENCY_MISSING_RE = re.compile(
    r"(?i)\b("
    r"not\s+(?:listed|present|declared|included|found)|"
    r"missing\s+(?:from|in)|"
    r"absent\s+from|"
    r"not\s+in|"
    r"need(?:s)?\s+to\s+be\s+added|"
    r"add\s+(?:to|them\s+to)\s+(?:package\.json|requirements)|"
    r"will\s+lead\s+to\s+runtime\s+errors"
    r")\b"
)

# Import/include statements, in any language we are likely to review.
#
# Deliberately a keyword set rather than per-language parsers: the C-only
# #include check below could not see `use axum::http::HeaderMap` and reported a
# false missing-import on a Rust PR. Every language spells this differently but
# they all put it on its own line near the top of the file, so one line-shape
# regex covers C/C++/ObjC, Rust, Python, Go, Java, Kotlin, C#, JS/TS, Ruby, PHP
# without needing to know which language we are looking at.
_IMPORT_LINE_RE = re.compile(
    r"""^\s*(?:
          \#\s*(?:include|import)\b        # C, C++, Objective-C
        | (?:pub\s+)?use\b                 # Rust, PHP
        | extern\s+crate\b                 # Rust 2015
        | import\b                         # Python, Java, Go, JS/TS, Kotlin
        | from\s+\S+\s+import\b            # Python
        | using\b                          # C#, C++ using-declarations
        | require(?:_relative)?\b          # Ruby, Node
        | include_once\b | require_once\b  # PHP
    )""",
    re.VERBOSE,
)

# An import statement quoted inside the finding itself. When the model writes
# "add `use axum::http::HeaderMap;`" the suggestion names the symbol far more
# reliably than prose does, so prefer this over identifier heuristics.
_IMPORT_STMT_RE = re.compile(
    r"""(?:
          \#\s*(?:include|import)\s*[<"']([^>"']+)[>"']
        | (?:pub\s+)?use\s+([A-Za-z_][\w:.]*(?:::\{[^}]*\})?)
        | extern\s+crate\s+([A-Za-z_]\w*)
        | from\s+([A-Za-z_][\w.]*)\s+import\s+([A-Za-z_*][\w,\s]*)
        | import\s+([A-Za-z_][\w.:/]*)
        | using\s+([A-Za-z_][\w.]*)
    )""",
    re.VERBOSE,
)

# "X is used but not imported" / "missing import for X" / "unresolved symbol X".
_IMPORT_CLAIM_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:not|never|isn'?t|is\s+not|without)\b[^.;]{0,40}?\bimport(?:ed|s)?\b"
    r"|\bimport(?:ed|s)?\b[^.;]{0,40}?\b(?:missing|absent|not\s+present|required)\b"
    r"|\bmissing\s+(?:an?\s+)?(?:import|use\s+statement|using\s+directive)\b"
    r"|\b(?:add|needs?|requires?)\b[^.;]{0,40}?\b(?:import|use\s+statement)\b"
    r"|\bunresolved\s+(?:import|symbol|reference)\b"
    r"|\bnot\s+(?:in\s+scope|declared|brought\s+into\s+scope)\b"
    r")"
)

# "removing X breaks Y" / "X is deleted but still used at runtime".
#
# The Bedd-unwire batch was almost entirely this shape: a Dockerfile stage that
# built an unused binary was deleted, and the model asserted the runtime would
# now fail. Whether anything still reaches for the removed thing is a question
# a grep answers exactly, in both directions — no callers disproves the claim,
# and a surviving caller is a real bug worth keeping.
_REMOVAL_CLAIM_RE = re.compile(
    r"(?i)\b(?:"
    r"remov(?:e|es|ed|ing|al)|delet(?:e|es|ed|ing)|dropp?(?:ed|ing)?|"
    r"no\s+longer\s+(?:present|available|built|installed|exists?|shipped)|"
    r"stripp?(?:ed|ing)|unwir(?:e|ed|ing)|elimina(?:te|ted|ting)"
    r")\b"
)
_REMOVAL_BREAKS_RE = re.compile(
    r"(?i)\b(?:"
    r"break(?:s|age)?|fail(?:s|ure|ing)?|crash(?:es|ing)?|"
    r"runtime\s+(?:error|failure)|will\s+not\s+start|won'?t\s+start|"
    r"missing\s+at\s+runtime|not\s+found\s+at\s+runtime|"
    r"still\s+(?:used|referenced|called|invoked|require[sd])|"
    r"depend(?:s|ent|encies)\s+on\s+it|production\s+(?:dies|breaks|outage)"
    r")\b"
)

# "this will not compile" / "breaks the build" / "signature mismatch".
#
# Cheapest possible disproof: CI already ran the real compiler on the PR head.
# If every check is green, a finding predicting a build failure at that SHA is
# wrong, and we know it for zero prompt tokens and without understanding the
# language. Only build claims — a green build says nothing about a memory leak.
_BUILD_CLAIM_RE = re.compile(
    r"(?i)\b(?:"
    r"compil(?:e|es|ing|ation)\s+(?:error|failure|fail|problem)"
    r"|(?:will\s+not|won'?t|fails?\s+to|does\s+not|doesn'?t|cannot|can'?t)\s+compil\w*"
    r"|(?:build|compile)\s+(?:error|failure|break(?:s|age)?)"
    r"|(?:will|would|could|may)\s+break\s+(?:the\s+)?(?:existing\s+)?"
    r"(?:build|compilation|route|router|call(?:er|s)?|signature|definition|configuration)"
    r"|signature\s+mismatch|type\s+mismatch|type\s+error"
    r"|does\s+not\s+(?:type[\s-]?check|typecheck)"
    r"|fails\s+to\s+build"
    r")\b"
)

# Words that follow "use"/"import"/"require" in ordinary English, so the
# statement regex above captures them as if they were symbols.
_IMPORT_PROSE_WORDS = frozenset({
    "statement", "statements", "directive", "directives", "declaration",
    "declarations", "clause", "clauses", "line", "lines", "block", "blocks",
    "it", "them", "this", "that", "these", "those", "the", "a", "an",
    "of", "for", "from", "in", "at", "to", "and", "or",
    # Connectives that land in the subject slot of "... but is not imported".
    "but", "however", "which", "also", "still", "yet", "so", "then",
    "there", "here", "they", "he", "she", "we", "you", "i", "who",
    "thus", "therefore", "though", "although", "while", "whereas",
    "import", "imports", "include", "includes", "use", "uses", "using",
    "crate", "crates", "module", "modules", "package", "packages",
    "type", "types", "trait", "traits", "class", "classes",
})

# The subject of the claim: "`timezone` is referenced but is not imported",
# "HeaderMap is used as an extractor". Catches bare lowercase identifiers that
# the code-shape regex below cannot distinguish from prose.
_IMPORT_SUBJECT_RE = re.compile(
    r"[`'\"]?\b([A-Za-z_][\w:.]*)\b[`'\"]?\s+(?:is|are|was|were)\s+"
    r"(?:\w+\s+){0,3}?(?:used|referenced|called|invoked|needed|required|missing"
    r"|not\s+(?:imported|declared|in\s+scope))"
)

# Symbols the model put in backticks or quotes.
_QUOTED_IDENT_RE = re.compile(r"[`'\"]([A-Za-z_][\w:.]{1,60})[`'\"]")

# Identifiers that read like code rather than prose: CamelCase, snake_case,
# or a qualified path. Used only as a fallback when the finding does not quote
# an import statement outright.
_CODE_IDENT_RE = re.compile(
    r"\b([A-Za-z_][\w]*(?:(?:::|\.)[A-Za-z_]\w*)+"      # foo::Bar, foo.Bar
    r"|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+"                # CamelCase
    r"|[a-z0-9]+_[a-z0-9_]+)\b"                          # snake_case
)

_CPP_SUFFIXES = frozenset({
    ".c", ".cc", ".cpp", ".cxx", ".c++",
    ".h", ".hpp", ".hh", ".hxx", ".h++",
    ".inl", ".ipp", ".m", ".mm",
})

# "missing #include", "header not included", "requires <fcntl.h>" style claims.
_INCLUDE_CLAIM_RE = re.compile(
    r"(?i)(?:"
    r"#\s*include"
    r"|\b(?:missing|absent|lacks?|lacking|needs?|requires?|add|without|no)\b"
    r"[^.;]{0,48}?\b(?:include|header)\b"
    r"|\b(?:include|header)\b[^.;]{0,48}?"
    r"\b(?:missing|absent|required|needed|not\s+(?:included|present|found|declared))\b"
    r")"
)

# Header spellings we can positively check: <foo.h>, "foo/bar.hpp", <cstring>.
_INCLUDE_DIRECTIVE_RE = re.compile(r"#\s*include\s*[<\"]([^>\"]+)[>\"]")
_HEADER_PATH_RE = re.compile(
    r"[<\"'`]?\b([A-Za-z0-9_][A-Za-z0-9_./+-]*\.(?:h|hpp|hh|hxx|h\+\+|inl|ipp))\b[>\"'`]?"
)
_ANGLE_HEADER_RE = re.compile(r"<([A-Za-z][A-Za-z0-9_./+-]*)>")

# "the directory/path X does not exist / is missing / is not on the include path"
_PATH_CLAIM_RE = re.compile(
    r"(?i)\b("
    r"director(?:y|ies)|include\s+path|include\s+dir\w*|search\s+path|"
    r"target_include_directories|include_directories|"
    r"CMAKE_SOURCE_DIR|CMAKE_CURRENT_SOURCE_DIR|PROJECT_SOURCE_DIR|"
    r"CMAKE_CURRENT_LIST_DIR"
    r")\b"
)
_PATH_MISSING_RE = re.compile(
    r"(?i)("
    r"\bdoes\s+not\s+exist\b|\bdoesn't\s+exist\b|\bnon-?existent\b|"
    r"\bnot\s+(?:found|present|declared|configured|set|added|listed|reachable|under)\b|"
    r"\bcan(?:not|'t)\s+be\s+(?:found|resolved|located)\b|"
    r"\bunable\s+to\s+(?:find|resolve|locate)\b|"
    r"\bunresolved\b|\bcompile\s+error\b|"
    r"\bmissing\b|\babsent\b|\bnever\s+added\b|\bis\s+not\s+on\s+the\b"
    r")"
)
# CMake variables that expand to the repository root.
_CMAKE_ROOT_VARS = ("CMAKE_SOURCE_DIR", "PROJECT_SOURCE_DIR", "CMAKE_CURRENT_SOURCE_DIR")
_CMAKE_VAR_RE = re.compile(r"\$\{(\w+)\}/?")
_PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_${}./+-]*/[A-Za-z0-9_${}./+-]+")

# Prefer explicit code mentions first.
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_QUOTED_RE = re.compile(r"['\"]([^'\"]+)['\"]")
# Identifier-ish tokens (drop very short / common English words).
_IDENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]{2,}|[a-z_][a-z0-9_]{3,})\b")
_PKG_NAME_RE = re.compile(
    r"^(?:@[A-Za-z0-9._-]+/)?[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$"
)
# "the transformers library" / "package numpy" — unquoted prose mentions.
_PKG_CONTEXT_RE = re.compile(
    r"(?i)\b(?:the\s+)?([A-Za-z][\w.-]{1,40})\s+(?:library|package|module|dependency)\b"
    r"|\b(?:library|package|module|dependency)\s+([A-Za-z][\w.-]{1,40})\b"
)
_STOPWORDS = frozenset({
    "this", "that", "with", "from", "import", "class", "def", "return",
    "true", "false", "none", "null", "self", "alias", "before", "after",
    "undefined", "defined", "definition", "assignment", "raises", "error",
    "nameerror", "name", "symbol", "missing", "exists", "exist", "file",
    "line", "code", "logic", "reassigns", "reassign", "suggestion", "replace",
    "remove", "create", "single", "backward", "compatible", "module",
    "python", "issue", "should", "would", "could", "seems", "appears",
    "package", "json", "lock", "dependencies", "dependency", "npm", "install",
    "runtime", "errors", "explicitly", "added", "listed", "present",
    # Nouns that follow "dependency"/"package" in prose and would otherwise be
    # read as the package name itself ("the project's dependency manifests").
    "manifest", "manifests", "list", "lists", "declaration", "declarations",
    "specification", "specifications", "version", "versions", "requirement",
    "requirements", "section", "sections", "entry", "entries", "block",
    # Qualifiers that precede "dependency" ("development/test dependency list").
    "test", "tests", "testing", "development", "dev", "optional", "build",
    "peer", "transitive", "external", "third", "party", "direct", "indirect",
})


def _collect_import_lines(source: str) -> list[str]:
    """Import statements, including the bodies of multi-line ones.

    Matching line-by-line kept only the first line of a wrapped import, so

        from app.schemas import (
            B2BDocumentType,
        )

    contributed ``from app.schemas import (`` and nothing else. Every symbol in
    the parenthesised body was invisible to the verifier, which then could not
    disprove "B2BDocumentType is never imported" even though the binding was
    right there. Continuation is tracked by bracket depth, which covers
    Python's parenthesised form, Rust's ``use foo::{A, B}`` and JS named
    imports without needing to know the language.
    """
    lines: list[str] = []
    depth = 0
    continued = False

    for line in source.splitlines():
        starts_import = bool(_IMPORT_LINE_RE.match(line))
        if not starts_import and depth <= 0 and not continued:
            continue

        lines.append(line)

        code = line.split("#", 1)[0].split("//", 1)[0]
        depth += code.count("(") + code.count("{") + code.count("[")
        depth -= code.count(")") + code.count("}") + code.count("]")
        depth = max(0, depth)
        # A trailing backslash continues the statement without any bracket.
        continued = code.rstrip().endswith("\\")

    return lines


@dataclass
class SuppressionRecord:
    file: str | None
    line: int | None
    message: str
    symbol: str | None
    reason: str


@dataclass
class VerificationReport:
    kept: list[ReviewIssue] = field(default_factory=list)
    suppressed: list[SuppressionRecord] = field(default_factory=list)

    @property
    def suppressed_count(self) -> int:
        return len(self.suppressed)


class ClaimVerifier:
    """Filter structurally impossible / falsely missing-dep findings on PR HEAD."""

    def __init__(
        self,
        indexer: SymbolIndexer | None = None,
        *,
        build_green: bool | None = None,
        build_sha: str = "",
    ):
        self.indexer = indexer or SymbolIndexer()
        self._tracked_cache: dict[Path, tuple[set[str], dict[str, str]]] = {}
        self._reference_cache: dict[tuple[Path, str], list[str] | None] = {}
        # CI's verdict on the PR head. None = unknown, and unknown never
        # suppresses anything. Fetched by the caller so this class stays
        # network-free and testable.
        self.build_green = build_green
        self.build_sha = build_sha

    def verify_result(
        self,
        result: ReviewResult,
        *,
        repo_root: Path,
        changed_paths: list[str] | None = None,
        indexes: list[FileSymbolIndex] | None = None,
    ) -> ReviewResult:
        report = self.verify_issues(
            result.issues,
            repo_root=repo_root,
            changed_paths=changed_paths,
            indexes=indexes,
        )
        if not report.suppressed:
            return result

        logger.info(
            f"ClaimVerifier suppressed {report.suppressed_count} "
            f"finding(s) (high-confidence disproven only)"
        )
        for record in report.suppressed:
            logger.info(
                f"  suppressed: {record.file}:{record.line} "
                f"symbol={record.symbol!r} — {record.reason}"
            )

        new_issues = report.kept
        meta = dict(result.routing_meta or {})
        meta["claim_verifier"] = {
            "suppressed": report.suppressed_count,
            "details": [
                {
                    "file": r.file,
                    "line": r.line,
                    "symbol": r.symbol,
                    "reason": r.reason,
                    "message": r.message[:200],
                }
                for r in report.suppressed
            ],
        }

        return ReviewResult(
            summary=result.summary,
            issues=new_issues,
            recommendations=result.recommendations,
            score=compute_score_from_issues(new_issues),
            latency_ms=result.latency_ms,
            model=result.model,
            tokens_used=result.tokens_used,
            review_type=result.review_type,
            parse_warning=result.parse_warning,
            routing_meta=meta,
        )

    def verify_issues(
        self,
        issues: list[ReviewIssue],
        *,
        repo_root: Path,
        changed_paths: list[str] | None = None,
        indexes: list[FileSymbolIndex] | None = None,
    ) -> VerificationReport:
        report = VerificationReport()
        if not issues:
            return report

        declared = load_declared_dependencies(repo_root)
        index_by_path = self._load_indexes(repo_root, changed_paths, indexes, issues)

        for issue in issues:
            if self._looks_path_claim(issue):
                confidence, symbol, reason = self._evaluate_path(issue, repo_root)
                if confidence == "disproven":
                    report.suppressed.append(
                        SuppressionRecord(
                            file=issue.file,
                            line=issue.line,
                            message=issue.message,
                            symbol=symbol,
                            reason=reason,
                        )
                    )
                else:
                    report.kept.append(issue)
                continue

            # Before the import/build checks: "you removed the bedd binary and
            # the runtime will fail" is not a missing-import claim, and none of
            # the checks below can settle it.
            if self._looks_removal_claim(issue):
                confidence, symbol, reason = self._evaluate_removal(
                    issue, repo_root, changed_paths
                )
                if confidence == "disproven":
                    report.suppressed.append(
                        SuppressionRecord(
                            file=issue.file,
                            line=issue.line,
                            message=issue.message,
                            symbol=symbol,
                            reason=reason,
                        )
                    )
                else:
                    report.kept.append(issue)
                continue

            if self._looks_include_claim(issue):
                confidence, symbol, reason = self._evaluate_include(issue, repo_root)
                if confidence == "disproven":
                    report.suppressed.append(
                        SuppressionRecord(
                            file=issue.file,
                            line=issue.line,
                            message=issue.message,
                            symbol=symbol,
                            reason=reason,
                        )
                    )
                else:
                    report.kept.append(issue)
                continue

            # Generic import check, after the C/C++ header path above: that one
            # understands header basenames, this one covers every other language.
            if self._looks_import_claim(issue):
                confidence, symbol, reason = self._evaluate_import(issue, repo_root)
                if confidence == "disproven":
                    report.suppressed.append(
                        SuppressionRecord(
                            file=issue.file,
                            line=issue.line,
                            message=issue.message,
                            symbol=symbol,
                            reason=reason,
                        )
                    )
                    continue
                # Not disproven as an import claim — it may still be a build
                # claim CI can settle, so fall through instead of keeping here.

            if self._looks_build_claim(issue):
                confidence, symbol, reason = self._evaluate_build(issue)
                if confidence == "disproven":
                    report.suppressed.append(
                        SuppressionRecord(
                            file=issue.file,
                            line=issue.line,
                            message=issue.message,
                            symbol=symbol,
                            reason=reason,
                        )
                    )
                else:
                    report.kept.append(issue)
                continue

            if self._looks_dependency_claim(issue):
                confidence, symbol, reason = self._evaluate_dependency(issue, declared)
                if confidence == "disproven":
                    report.suppressed.append(
                        SuppressionRecord(
                            file=issue.file,
                            line=issue.line,
                            message=issue.message,
                            symbol=symbol,
                            reason=reason,
                        )
                    )
                else:
                    report.kept.append(issue)
                continue

            # An import claim reaches here only when the file's own import
            # lines did not settle it. The symbol index is a second, independent
            # source of the same fact — it records multi-line ImportFrom aliases
            # as bindings — and phrasings like "referenced but not imported"
            # never match _STRUCTURAL_RE, so without this they were kept on the
            # strength of one check that had already come back uncertain.
            if not self._looks_structural(issue) and not self._looks_import_claim(issue):
                report.kept.append(issue)
                continue

            confidence, symbol, reason = self._evaluate(issue, index_by_path)
            if confidence == "disproven":
                report.suppressed.append(
                    SuppressionRecord(
                        file=issue.file,
                        line=issue.line,
                        message=issue.message,
                        symbol=symbol,
                        reason=reason,
                    )
                )
            else:
                report.kept.append(issue)

        return report

    def _load_indexes(
        self,
        repo_root: Path,
        changed_paths: list[str] | None,
        indexes: list[FileSymbolIndex] | None,
        issues: list[ReviewIssue],
    ) -> dict[str, FileSymbolIndex]:
        if indexes is not None:
            return {idx.path: idx for idx in indexes}

        paths: list[str] = list(changed_paths or [])
        for issue in issues:
            if issue.file and issue.file not in paths:
                paths.append(issue.file)
        return {idx.path: idx for idx in self.indexer.index_files(repo_root, paths)}

    def _looks_structural(self, issue: ReviewIssue) -> bool:
        blob = " ".join(
            part for part in (issue.message, issue.suggestion or "", issue.rule or "") if part
        )
        return bool(_STRUCTURAL_RE.search(blob))

    def _looks_dependency_claim(self, issue: ReviewIssue) -> bool:
        blob = " ".join(
            part for part in (issue.message, issue.suggestion or "", issue.rule or "") if part
        )
        return bool(
            _DEPENDENCY_MANIFEST_RE.search(blob) and _DEPENDENCY_MISSING_RE.search(blob)
        )

    def _looks_path_claim(self, issue: ReviewIssue) -> bool:
        blob = " ".join(
            part for part in (issue.message, issue.suggestion or "", issue.rule or "") if part
        )
        return bool(_PATH_CLAIM_RE.search(blob) and _PATH_MISSING_RE.search(blob))

    def _evaluate_path(
        self,
        issue: ReviewIssue,
        repo_root: Path,
    ) -> tuple[str, str | None, str]:
        """Disprove a "path X does not exist" claim against tracked paths on HEAD."""
        candidates = self._extract_paths(issue)
        if not candidates:
            return "uncertain", None, "could not extract a path from finding"

        tracked_dirs, tracked_by_name = self._tracked_paths(repo_root)
        if not tracked_dirs and not tracked_by_name:
            return "uncertain", candidates[0], "could not list tracked paths on HEAD"

        for candidate in candidates:
            rel = candidate.strip("/")
            if not rel:
                continue
            if (repo_root / rel).exists():
                return (
                    "disproven",
                    candidate,
                    f"{candidate} exists on PR HEAD",
                )
            # The model often gets the parent wrong while naming the right
            # directory — ${CMAKE_SOURCE_DIR}/internal_headers when the tree
            # actually has src/internal_headers.
            name = PurePosixPath(rel).name
            actual = tracked_by_name.get(name)
            if actual:
                return (
                    "disproven",
                    candidate,
                    f"{name} exists on PR HEAD at {actual} "
                    f"(claim named the wrong parent directory)",
                )

        return "uncertain", candidates[0], "claimed paths not found on HEAD"

    def _extract_paths(self, issue: ReviewIssue) -> list[str]:
        blob = " ".join(
            part for part in (issue.message, issue.suggestion or "") if part
        )
        found: list[str] = []

        def add(raw: str) -> None:
            # ${CMAKE_SOURCE_DIR}/internal_headers → internal_headers
            expanded = _CMAKE_VAR_RE.sub(
                lambda m: "" if m.group(1) in _CMAKE_ROOT_VARS else m.group(0),
                raw,
            ).strip()
            expanded = expanded.strip("\"'`()").strip()
            if not expanded or "${" in expanded:
                return
            if expanded not in found:
                found.append(expanded)

        for pattern in (_BACKTICK_RE, _QUOTED_RE):
            for match in pattern.finditer(blob):
                for token in _PATH_TOKEN_RE.findall(match.group(1)):
                    add(token)
        if found:
            return found

        for token in _PATH_TOKEN_RE.findall(blob):
            add(token)
        return found

    # ---- removal claims, settled by searching for surviving references ------

    def _looks_removal_claim(self, issue: ReviewIssue) -> bool:
        blob = " ".join(
            part for part in (issue.message, issue.suggestion or "", issue.rule or "") if part
        )
        return bool(_REMOVAL_CLAIM_RE.search(blob) and _REMOVAL_BREAKS_RE.search(blob))

    def _evaluate_removal(
        self,
        issue: ReviewIssue,
        repo_root: Path,
        changed_paths: list[str] | None,
    ) -> tuple[str, str | None, str]:
        """Disprove "removing X breaks things" when nothing references X.

        Searches every tracked file, not just source extensions: the caller that
        matters is as likely to live in a compose file, a shell script or a
        JSON manifest as in a .py. Silence is the disproof; a surviving
        reference keeps the finding, which is how the one real leftover in this
        batch (scripts/smoke-test.js calling prisma.embedding) would have been
        caught rather than invented around.
        """
        symbols = self._extract_removal_symbols(issue)
        if not symbols:
            return "uncertain", None, "could not extract a removed name from finding"

        excluded = {p for p in (changed_paths or [])}
        if issue.file:
            excluded.add(issue.file)

        checked: list[str] = []
        for symbol in symbols:
            hits = self._find_references(repo_root, symbol)
            if hits is None:
                return "uncertain", symbol, "reference search unavailable"

            surviving = [h for h in hits if h not in excluded]
            if surviving:
                shown = ", ".join(surviving[:3])
                return (
                    "uncertain",
                    symbol,
                    f"{symbol} is still referenced on PR HEAD by {shown}",
                )
            checked.append(symbol)

        joined = ", ".join(checked)
        return (
            "disproven",
            checked[0],
            f"no tracked file outside the diff references {joined} on PR HEAD",
        )

    def _extract_removal_symbols(self, issue: ReviewIssue) -> list[str]:
        """Names the finding says were removed: backticked/quoted first."""
        blob = " ".join(part for part in (issue.message, issue.suggestion or "") if part)
        found: list[str] = []

        def add(name: str | None) -> None:
            if not name:
                return
            name = name.strip().strip("`'\"<>(),;:").strip()
            # A bare path reduces to its basename: the finding may cite
            # /usr/local/bin/bedd while callers just say `bedd`.
            if "/" in name:
                name = PurePosixPath(name).name
            if len(name) < 3 or name.lower() in _IMPORT_PROSE_WORDS:
                return
            if name.lower() in _STOPWORDS:
                return
            if name not in found:
                found.append(name)

        for match in _BACKTICK_RE.finditer(blob):
            add(match.group(1))
        for match in _QUOTED_RE.finditer(blob):
            add(match.group(1))
        if found:
            return found[:3]

        # ENV_VAR_STYLE and CamelCase/snake_case identifiers read as deliberate
        # names; ordinary prose does not.
        for match in re.finditer(r"\b([A-Z][A-Z0-9_]{2,}|[A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)+)\b", blob):
            add(match.group(1))
        for match in _CODE_IDENT_RE.finditer(blob):
            add(match.group(1))
        return found[:3]

    def _find_references(self, repo_root: Path, symbol: str) -> list[str] | None:
        """Tracked files containing `symbol`. None when the search cannot run."""
        key = (repo_root, symbol)
        if key in self._reference_cache:
            return self._reference_cache[key]

        try:
            proc = subprocess.run(
                [
                    "git", "-C", str(repo_root), "grep",
                    "--fixed-strings", "--name-only", "--ignore-case",
                    "-e", symbol,
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
            logger.debug(f"Reference search failed for {symbol}: {exc}")
            self._reference_cache[key] = None
            return None

        # git grep exits 1 for "no matches", which is a real answer, not a
        # failure. Anything else means the search itself did not run.
        if proc.returncode not in (0, 1):
            logger.debug(
                f"Reference search for {symbol} exited {proc.returncode}: "
                f"{proc.stderr.strip()[:120]}"
            )
            self._reference_cache[key] = None
            return None

        hits = [line for line in proc.stdout.splitlines() if line]
        self._reference_cache[key] = hits
        return hits

    def _tracked_paths(self, repo_root: Path) -> tuple[set[str], dict[str, str]]:
        """Return (directory set, basename → first matching path) for HEAD."""
        cached = self._tracked_cache.get(repo_root)
        if cached is not None:
            return cached

        dirs: set[str] = set()
        by_name: dict[str, str] = {}
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo_root), "ls-files"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            rels = proc.stdout.splitlines()
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
            logger.debug(f"Path verifier could not list tracked files: {exc}")
            self._tracked_cache[repo_root] = (dirs, by_name)
            return dirs, by_name

        for rel in rels:
            if not rel:
                continue
            by_name.setdefault(PurePosixPath(rel).name, rel)
            parent = PurePosixPath(rel).parent
            while str(parent) not in (".", ""):
                dirs.add(str(parent))
                by_name.setdefault(parent.name, str(parent))
                parent = parent.parent

        self._tracked_cache[repo_root] = (dirs, by_name)
        return dirs, by_name

    def _looks_include_claim(self, issue: ReviewIssue) -> bool:
        if issue.file and Path(issue.file).suffix.lower() not in _CPP_SUFFIXES:
            return False
        blob = " ".join(
            part for part in (issue.message, issue.suggestion or "", issue.rule or "") if part
        )
        return bool(_INCLUDE_CLAIM_RE.search(blob))

    def _evaluate_include(
        self,
        issue: ReviewIssue,
        repo_root: Path,
    ) -> tuple[str, str | None, str]:
        """Disprove a missing-#include claim by reading the file's own includes."""
        if not issue.file:
            return "uncertain", None, "no file on finding"

        claimed = self._extract_headers(issue)
        if not claimed:
            return "uncertain", None, "could not extract a header name from finding"

        present = self._read_includes(repo_root, issue.file)
        if present is None:
            return "uncertain", claimed[0], f"could not read {issue.file} on HEAD"
        if not present:
            return "uncertain", claimed[0], f"no #include directives found in {issue.file}"

        # Match on the full path first, then bare filename — <sys/mman.h> cited
        # as `mman.h` should still count as present.
        by_base: dict[str, str] = {}
        for inc in present:
            by_base.setdefault(PurePosixPath(inc).name.lower(), inc)

        found: list[str] = []
        missing: list[str] = []
        for header in claimed:
            key = header.lower()
            if key in {p.lower() for p in present}:
                found.append(header)
            elif PurePosixPath(header).name.lower() in by_base:
                found.append(by_base[PurePosixPath(header).name.lower()])
            else:
                missing.append(header)

        if found and not missing:
            joined = ", ".join(dict.fromkeys(found))
            return (
                "disproven",
                found[0],
                f"{joined} already included in {issue.file} on PR HEAD "
                f"(outside the diff window — diff-only false positive)",
            )
        if found and missing:
            return (
                "uncertain",
                missing[0],
                f"partial: {', '.join(found)} included; {', '.join(missing)} not found",
            )
        return "uncertain", missing[0], "claimed headers not found in the file's include block"

    def _extract_headers(self, issue: ReviewIssue) -> list[str]:
        blob = " ".join(
            part for part in (issue.message, issue.suggestion or "") if part
        )
        found: list[str] = []

        def add(name: str) -> None:
            name = name.strip().strip("<>\"'`")
            if name and name not in found:
                found.append(name)

        # Strongest signal: the finding literally spells the directive.
        for match in _INCLUDE_DIRECTIVE_RE.finditer(blob):
            add(match.group(1))
        if found:
            return found

        for match in _HEADER_PATH_RE.finditer(blob):
            add(match.group(1))
        # Extension-less C++ std headers (<vector>, <cstring>) only in angles,
        # so we don't mistake prose for a header name.
        for match in _ANGLE_HEADER_RE.finditer(blob):
            add(match.group(1))

        return found

    def _read_includes(self, repo_root: Path, rel_path: str) -> set[str] | None:
        path = (repo_root / rel_path).resolve()
        try:
            path.relative_to(repo_root.resolve())
        except ValueError:
            logger.debug(f"Include scan refused path outside repo: {rel_path}")
            return None
        if not path.is_file():
            return None
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            logger.debug(f"Include scan read failed for {rel_path}: {exc}")
            return None
        return {m.group(1) for m in _INCLUDE_DIRECTIVE_RE.finditer(source)}

    # ---- generic (language-agnostic) missing-import claims -----------------

    def _looks_import_claim(self, issue: ReviewIssue) -> bool:
        blob = " ".join(
            part for part in (issue.message, issue.suggestion or "", issue.rule or "") if part
        )
        return bool(_IMPORT_CLAIM_RE.search(blob))

    def _evaluate_import(
        self,
        issue: ReviewIssue,
        repo_root: Path,
    ) -> tuple[str, str | None, str]:
        """Disprove "X is not imported" by reading the file's own import lines.

        Language-agnostic on purpose. A symbol that is genuinely un-imported
        still appears at its use sites, so presence in the file is not enough —
        the symbol has to appear on a line that is *shaped* like an import.
        """
        if not issue.file:
            return "uncertain", None, "no file on finding"

        symbols = self._extract_import_symbols(issue)
        if not symbols:
            return "uncertain", None, "could not extract a symbol name from finding"

        lines = self._read_import_lines(repo_root, issue.file)
        if lines is None:
            return "uncertain", symbols[0], f"could not read {issue.file} on HEAD"
        if not lines:
            return "uncertain", symbols[0], f"no import statements found in {issue.file}"

        blob = "\n".join(lines)
        found: list[str] = []
        missing: list[str] = []
        for symbol in symbols:
            # Match the last path segment too: a claim about `axum::http::HeaderMap`
            # is satisfied by `use axum::http::{HeaderMap, StatusCode};`.
            leaf = next(
                (part for part in reversed(re.split(r"::|\.|/", symbol)) if part), ""
            )
            # An empty or one-character leaf would match almost any import line,
            # so it is not evidence of anything.
            if len(leaf) < 2:
                continue
            pattern = re.compile(rf"(?<![\w]){re.escape(leaf)}(?![\w])")
            if pattern.search(blob):
                found.append(symbol)
            else:
                missing.append(symbol)

        if not found and not missing:
            return "uncertain", symbols[0], "no usable symbol name in finding"

        if found and not missing:
            joined = ", ".join(dict.fromkeys(found))
            return (
                "disproven",
                found[0],
                f"{joined} already imported in {issue.file} on PR HEAD "
                f"(outside the diff window — diff-only false positive)",
            )
        if found and missing:
            return (
                "uncertain",
                missing[0],
                f"partial: {', '.join(found)} imported; {', '.join(missing)} not found",
            )
        return "uncertain", missing[0], "claimed symbols not found in the file's imports"

    def _extract_import_symbols(self, issue: ReviewIssue) -> list[str]:
        blob = " ".join(part for part in (issue.message, issue.suggestion or "") if part)
        found: list[str] = []

        def add(name: str | None) -> None:
            if not name:
                return
            name = name.strip().strip("<>\"'`,;{}.").strip()
            # "the necessary use statement" makes `use ...` match plain prose;
            # a bare English word is never the symbol we are looking for.
            if not name or name.lower() in _IMPORT_PROSE_WORDS:
                return
            if name not in found:
                found.append(name)

        # Strongest signal: the finding quotes an import statement outright.
        for match in _IMPORT_STMT_RE.finditer(blob):
            for group in match.groups():
                add(group)
        if found:
            return found

        # Next: the grammatical subject of the claim, then quoted symbols. Both
        # catch bare lowercase identifiers ("timezone") that are indistinguishable
        # from prose by shape alone.
        for match in _IMPORT_SUBJECT_RE.finditer(blob):
            add(match.group(1))
        for match in _QUOTED_IDENT_RE.finditer(blob):
            add(match.group(1))
        if found:
            return found[:4]

        # Last resort: identifiers that look like code, not prose.
        for match in _CODE_IDENT_RE.finditer(blob):
            token = match.group(1)
            if token.lower() in _STOPWORDS:
                continue
            add(token)
        return found[:4]

    def _read_import_lines(self, repo_root: Path, rel_path: str) -> list[str] | None:
        path = (repo_root / rel_path).resolve()
        try:
            path.relative_to(repo_root.resolve())
        except ValueError:
            logger.debug(f"Import scan refused path outside repo: {rel_path}")
            return None
        if not path.is_file():
            return None
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            logger.debug(f"Import scan read failed for {rel_path}: {exc}")
            return None
        return _collect_import_lines(source)

    # ---- build-failure claims, disproved by CI ------------------------------

    def _looks_build_claim(self, issue: ReviewIssue) -> bool:
        blob = " ".join(
            part for part in (issue.message, issue.suggestion or "", issue.rule or "") if part
        )
        return bool(_BUILD_CLAIM_RE.search(blob))

    def _evaluate_build(self, issue: ReviewIssue) -> tuple[str, str | None, str]:
        """Disprove a predicted build failure using CI's verdict on the PR head.

        ``self.build_green`` is None whenever we could not establish a verdict
        (checks still running, no checks configured, unknown head SHA). Only a
        definite green result disproves anything.
        """
        if self.build_green is None:
            return "uncertain", None, "no CI verdict available for the PR head"
        if not self.build_green:
            return "uncertain", None, "CI is not green — build claim may well be right"
        where = f" at {self.build_sha[:7]}" if self.build_sha else ""
        return (
            "disproven",
            None,
            f"CI build is green{where} — a predicted compile/build failure is "
            f"contradicted by the real compiler",
        )

    def _evaluate_dependency(
        self,
        issue: ReviewIssue,
        declared: set[str],
    ) -> tuple[str, str | None, str]:
        packages = self._extract_package_names(issue)
        if not packages:
            return "uncertain", None, "could not extract package name from finding"
        if not declared:
            return "uncertain", packages[0], "no dependency manifests readable on HEAD"

        declared_l = {n.lower(): n for n in declared}
        present: list[str] = []
        missing: list[str] = []
        for pkg in packages:
            key = pkg.lower()
            alt = pkg.replace("_", "-").lower()
            if key in declared_l or alt in declared_l or pkg in declared:
                present.append(declared_l.get(key) or declared_l.get(alt) or pkg)
            else:
                missing.append(pkg)

        if present and not missing:
            joined = ", ".join(present)
            return (
                "disproven",
                present[0],
                f"{joined} declared on PR HEAD (diff-only false positive)",
            )
        if present and missing:
            # Partial: keep the finding — some claimed packages really absent.
            return (
                "uncertain",
                missing[0],
                f"partial: {', '.join(present)} present; {', '.join(missing)} not found",
            )
        return "uncertain", missing[0], "claimed packages not found on HEAD manifests"

    def _extract_package_names(self, issue: ReviewIssue) -> list[str]:
        blob = " ".join(
            part for part in (issue.message, issue.suggestion or "") if part
        )
        found: list[str] = []
        for pattern in (_BACKTICK_RE, _QUOTED_RE):
            for match in pattern.finditer(blob):
                raw = match.group(1).strip()
                # Drop path-like mentions; allow scoped npm @scope/name.
                if "/" in raw and not raw.startswith("@"):
                    continue
                if raw.lower() in {
                    "package.json",
                    "package-lock.json",
                    "requirements.txt",
                    "pyproject.toml",
                    "npm install",
                }:
                    continue
                # Apostrophes in prose ("the project's dependency list") look
                # like quote delimiters, so a "quoted" span containing spaces
                # is almost certainly a sentence fragment, not a package name.
                if len(raw.split()) > 1:
                    continue
                name = raw
                if len(name) < 2 or not _PKG_NAME_RE.match(name):
                    continue
                if name.lower() in _STOPWORDS:
                    continue
                if name not in found:
                    found.append(name)
        if found:
            return found

        # Nothing quoted — fall back to "the transformers library" / "package
        # numpy" phrasing, which is how models write it in prose.
        for match in _PKG_CONTEXT_RE.finditer(blob):
            name = (match.group(1) or match.group(2) or "").strip()
            if not name or name.lower() in _STOPWORDS:
                continue
            if not _PKG_NAME_RE.match(name):
                continue
            if name not in found:
                found.append(name)
        return found

    def _evaluate(
        self,
        issue: ReviewIssue,
        index_by_path: dict[str, FileSymbolIndex],
    ) -> tuple[str, str | None, str]:
        """Return (confidence, symbol, reason). confidence ∈ {disproven, uncertain}."""
        if not issue.file:
            return "uncertain", None, "no file on finding"

        index = index_by_path.get(issue.file)
        if index is None:
            # try basename match for path normalization quirks
            for path, candidate in index_by_path.items():
                if path.endswith(issue.file) or issue.file.endswith(path):
                    index = candidate
                    break
        if index is None:
            return "uncertain", None, "no symbol index for file"

        symbols = self._extract_symbols(issue)
        if not symbols:
            return "uncertain", None, "could not extract symbol from finding"

        cited = issue.line
        if cited is None:
            # Without a line we only suppress if the symbol clearly exists
            # somewhere at module scope — still positive existence, but weaker
            # for order claims; keep conservative: require existence only when
            # message is clearly "undefined/not defined" not "before defined".
            blob = (issue.message or "").lower()
            order_claim = "before" in blob and "defin" in blob
            if order_claim:
                return "uncertain", symbols[0], "order claim without cited line"

            for symbol in symbols:
                binding = index.first_binding(symbol)
                if binding is not None:
                    return (
                        "disproven",
                        symbol,
                        f"{symbol} is bound at module level (@{binding.line}, {binding.kind})",
                    )
            return "uncertain", symbols[0], "no module-level binding found"

        for symbol in symbols:
            binding = index.first_binding(symbol)
            if binding is None:
                continue

            # Cited line *is* the definition of the allegedly missing symbol.
            if binding.line == cited:
                return (
                    "disproven",
                    symbol,
                    f"{symbol} is defined at cited line {cited} ({binding.kind})",
                )

            # Confidence rule: must exist *before* the cited location.
            if binding.line < cited:
                return (
                    "disproven",
                    symbol,
                    f"{symbol} bound at line {binding.line} ({binding.kind}) "
                    f"before cited line {cited}",
                )

        return "uncertain", symbols[0], "could not positively disprove finding"

    def _extract_symbols(self, issue: ReviewIssue) -> list[str]:
        blob = " ".join(
            part for part in (issue.message, issue.suggestion or "") if part
        )
        found: list[str] = []

        for pattern in (_BACKTICK_RE, _QUOTED_RE):
            for match in pattern.finditer(blob):
                name = match.group(1).split(".")[-1]
                if name.lower() not in _STOPWORDS and name not in found:
                    found.append(name)

        if found:
            return found

        for match in _IDENT_RE.finditer(blob):
            name = match.group(1)
            if name.lower() in _STOPWORDS:
                continue
            if name not in found:
                found.append(name)

        return found
