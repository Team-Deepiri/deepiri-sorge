"""Conservative post-LLM claim verifier for structural / dependency false positives.

Policy (confidence rule):
  Suppress ONLY when the verifier can positively establish that the claim is
  false against the PR-head checkout (``confidence == "disproven"``). Otherwise
  preserve the original finding (``uncertain`` / ``confirmed``).

  1. Structural / NameError claims → symbol index (bound before cited line).
  2. Missing-dependency claims → package.json / requirements / pyproject on HEAD.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

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
_DEPENDENCY_MANIFEST_RE = re.compile(
    r"(?i)\b("
    r"package\.json|package-lock\.json|npm|yarn\.lock|pnpm-lock|"
    r"requirements(?:\.txt)?|pyproject\.toml|poetry\.lock|Pipfile(?:\.lock)?"
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

# Prefer explicit code mentions first.
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_QUOTED_RE = re.compile(r"['\"]([^'\"]+)['\"]")
# Identifier-ish tokens (drop very short / common English words).
_IDENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]{2,}|[a-z_][a-z0-9_]{3,})\b")
_PKG_NAME_RE = re.compile(
    r"^(?:@[A-Za-z0-9._-]+/)?[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$"
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
                name = raw.split()[0]
                if not _PKG_NAME_RE.match(name):
                    continue
                if name.lower() in _STOPWORDS:
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
