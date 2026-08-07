"""Conservative post-LLM claim verifier for structural / dependency false positives.

Policy (confidence rule):
  Suppress ONLY when the verifier can positively establish that the claim is
  false against the PR-head checkout (``confidence == "disproven"``). Otherwise
  preserve the original finding (``uncertain`` / ``confirmed``).

  1. Structural / NameError claims → symbol index (bound before cited line).
  2. Missing-dependency claims → package.json / requirements / pyproject on HEAD.
  3. Missing-``#include`` claims → the C/C++ file's own include block on HEAD.
  4. "Path X does not exist" claims → the tracked file/directory list on HEAD.

Rationale for (3): a diff only shows changed hunks plus a few lines of context.
An ``#include`` above the first hunk is invisible to the model, so "you use
open() without including <fcntl.h>" is a reasonable inference from an
incomplete picture. Checking the file on disk costs zero prompt tokens, where
widening the diff window would cost them on every chunk.
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

    def __init__(self, indexer: SymbolIndexer | None = None):
        self.indexer = indexer or SymbolIndexer()
        self._tracked_cache: dict[Path, tuple[set[str], dict[str, str]]] = {}

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

            if not self._looks_structural(issue):
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
