"""Diff-anchored resonance — token- and CPU-efficient repository context.

Strategy (fast + cheap):
1. Extract a small set of high-signal anchors from added diff lines only.
2. List candidate files via ``git ls-files`` (respects .gitignore, one subprocess).
3. Search only utility paths + same package as changed files; stop early on budget.
4. Emit compact one-line evidence cards (path:line | signature), not multi-line snippets.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from loguru import logger

from bot.config import RepoContextConfig
from bot.diff_parser import ParsedDiff

SKIP_DIR_NAMES = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
    ".tox",
    "htmlcov",
    "coverage",
}

UTILITY_PATH_SEGMENTS = frozenset({
    "utils",
    "util",
    "common",
    "lib",
    "libs",
    "helpers",
    "shared",
    "internal",
    "core",
})

SOURCE_EXTENSIONS = frozenset({
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".rb",
    ".php",
    # C / C++ — headers included so the index can surface real header paths.
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".hh",
    ".hxx",
})

IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))"
)
DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)")
CLASS_RE = re.compile(r"^\s*class\s+(\w+)")
# C/C++: header name from a changed #include, and class/struct/namespace decls.
INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]")
CPP_TYPE_RE = re.compile(r"^\s*(?:class|struct|namespace|enum(?:\s+class)?)\s+(\w+)")
TOKEN_RE = re.compile(r"[a-zA-Z_][\w]{2,}")
# Path-shaped tokens in a changed line: `cyrex-agi/bedd`, `/usr/local/bin/bedd`.
# Their segments are deliberate names rather than prose, so they earn a lower
# length floor than the bare-token scan — a removed binary is often a short
# word ("bedd") that would otherwise never become an anchor.
PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]*/[A-Za-z0-9_./-]+")
PATH_SEGMENT_MIN = 4

# Anchor weights — how deliberate a name is, which is what should survive the
# `max_anchors` cap. A declared symbol or a path segment is something somebody
# chose to write; a five-letter word lifted out of a build command is not.
WEIGHT_SYMBOL = 4.0
WEIGHT_CHANGED_FILE = 3.5
WEIGHT_PATH_SEGMENT = 3.0
WEIGHT_SYMBOL_PART = 2.0
WEIGHT_TOKEN = 1.0
SYMBOL_LINE_RE = re.compile(
    r"^(?:async\s+)?def\s+\w+"
    r"|^class\s+\w+"
    r"|^(?:export\s+)?(?:async\s+)?function\s+\w+"
    r"|^(?:template\s*<[^>]*>\s*)?(?:class|struct|namespace|enum(?:\s+class)?)\s+\w+"
    r"|^[\w:<>,\s\*&]+\s[\w:~]+\s*\([^;]*\)\s*(?:const\s*)?(?:noexcept\s*)?\{"
)

STOPWORDS = frozenset({
    "self", "true", "false", "none", "null", "return", "import", "from",
    "class", "def", "async", "await", "else", "elif", "with", "pass",
    "break", "continue", "raise", "yield", "lambda", "and", "not", "or",
    "try", "except", "finally", "for", "while", "in", "is", "as", "if",
    "print", "type", "str", "int", "list", "dict", "set", "tuple", "len",
    "open", "path", "file", "data", "value", "item", "name", "text", "line",
    # C/C++ keywords and ubiquitous std names — high frequency, zero signal.
    "const", "constexpr", "static", "inline", "extern", "struct", "union",
    "namespace", "template", "typename", "public", "private", "protected",
    "virtual", "override", "noexcept", "nullptr", "sizeof", "typedef",
    "unsigned", "signed", "size_t", "ssize_t", "uint8_t", "uint32_t",
    "uint64_t", "int32_t", "int64_t", "std", "string", "vector", "include",
    "define", "endif", "ifndef", "pragma", "delete", "new", "this", "auto",
    "using", "void", "bool", "char", "float", "double", "long", "short",
})


@dataclass
class CapabilityHit:
    path: str
    score: float
    reason: str
    signature: str
    line: int | None = None


@dataclass
class RepoContextPack:
    dependencies: list[str] = field(default_factory=list)
    hits: list[CapabilityHit] = field(default_factory=list)
    anchors: list[str] = field(default_factory=list)
    fingerprint: str = ""

    @property
    def text(self) -> str:
        return format_context_pack(self.hits, self.dependencies, self.anchors)


def format_context_pack(
    hits: list[CapabilityHit],
    dependencies: list[str],
    anchors: list[str],
) -> str:
    if not hits and not dependencies:
        return ""

    parts: list[str] = [
        "REPO_CONTEXT compact evidence — prefer these over new duplicate logic in the DIFF:",
    ]

    if dependencies:
        parts.append(f"deps: {', '.join(dependencies)}")

    if anchors:
        parts.append(f"anchors: {', '.join(anchors)}")

    for hit in hits:
        loc = f"{hit.path}:{hit.line}" if hit.line else hit.path
        sig = hit.signature.replace("\n", " ").strip()[:120]
        parts.append(f"reuse|{loc}|{sig} ({hit.reason})")

    return "\n".join(parts)


class RepoContextWeaver:
    def __init__(self, config: RepoContextConfig):
        self.config = config

    def weave(self, repo_root: Path, diff: ParsedDiff) -> RepoContextPack:
        if not self.config.enabled:
            return RepoContextPack()

        repo_root = repo_root.resolve()
        if not repo_root.is_dir():
            logger.warning(f"Repo root not found: {repo_root}")
            return RepoContextPack()

        changed = set(diff.files)
        anchors = self._rank_anchors(self._extract_anchors(diff))
        if not anchors:
            return RepoContextPack()

        dependencies = self._relevant_dependencies(repo_root, anchors)
        candidate_files = self._prioritize_files(
            self._list_tracked_files(repo_root),
            changed,
        )

        hits = self._search_candidates(
            repo_root=repo_root,
            candidate_files=candidate_files,
            anchors=anchors,
            changed_files=changed,
        )

        hits = self._apply_char_budget(hits, dependencies, anchors)

        pack = RepoContextPack(
            dependencies=dependencies,
            hits=hits,
            anchors=anchors,
            fingerprint=self._fingerprint(anchors, hits, dependencies),
        )
        logger.info(
            f"Repo context: {len(anchors)} anchors, {len(hits)} hits, "
            f"{len(pack.text)} chars, {len(candidate_files)} candidates"
        )
        return pack

    def _extract_anchors(self, diff: ParsedDiff) -> dict[str, float]:
        """Map anchor → weight, where weight reflects how deliberate the name is.

        Weight, not length, decides what survives ``max_anchors``. Ranking by
        length dropped "bedd" — the removed binary the whole PR was about — in
        favour of incidental prose like "release" and "builder".
        """
        anchors: dict[str, float] = {}

        def add(name: str, weight: float) -> None:
            lower = name.lower()
            if lower in STOPWORDS:
                return
            anchors[lower] = max(anchors.get(lower, 0.0), weight)

        for raw_line in diff.raw.splitlines():
            # Removed lines carry the same evidence value as added ones. A PR
            # that only deletes (an unwired Docker stage, a dropped Prisma
            # model) produced zero anchors while this read `+` alone, so
            # REPO_CONTEXT came back empty and the model reviewed the diff with
            # no repository evidence at all — the single largest source of
            # invented "this removal breaks production" findings.
            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                line = raw_line[1:]
            elif raw_line.startswith("-") and not raw_line.startswith("---"):
                line = raw_line[1:]
            else:
                continue

            # #include <sys/mman.h> → anchor on "mman", not the ".h" suffix.
            include_match = INCLUDE_RE.match(line)
            if include_match:
                header = include_match.group(1)
                add(Path(header).stem, WEIGHT_SYMBOL)

            for pattern in (IMPORT_RE, DEF_RE, CLASS_RE, CPP_TYPE_RE):
                match = pattern.match(line)
                if match:
                    for group in match.groups():
                        if group:
                            leaf = group.split(".")[-1]
                            add(leaf, WEIGHT_SYMBOL)
                            for part in re.split(r"[_-]", leaf):
                                if len(part) >= PATH_SEGMENT_MIN:
                                    add(part, WEIGHT_SYMBOL_PART)

            for token in TOKEN_RE.findall(line):
                if len(token) >= 5:
                    add(token, WEIGHT_TOKEN)

            for path_token in PATH_TOKEN_RE.findall(line):
                for segment in re.split(r"[/.]", path_token):
                    if len(segment) >= PATH_SEGMENT_MIN:
                        add(segment, WEIGHT_PATH_SEGMENT)

        # The changed files themselves name what the PR is about. A deletion in
        # `worker/bedd_bridge.py` should search for "bedd_bridge" even when the
        # removed body mentions it nowhere.
        for rel in diff.files:
            stem = PurePosixPath(rel).stem
            if len(stem) >= PATH_SEGMENT_MIN:
                add(stem, WEIGHT_CHANGED_FILE)
            for part in re.split(r"[_-]", stem):
                if len(part) >= PATH_SEGMENT_MIN:
                    add(part, WEIGHT_CHANGED_FILE)

        return anchors

    def _rank_anchors(self, anchors: dict[str, float]) -> list[str]:
        """Prefer deliberate names over incidental prose; cap for search cost."""
        ranked = sorted(
            anchors.items(),
            key=lambda item: (item[1], len(item[0])),
            reverse=True,
        )
        return [name for name, _ in ranked[: self.config.max_anchors]]

    def _list_tracked_files(self, repo_root: Path) -> list[str]:
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo_root), "ls-files"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            files = []
            for rel in proc.stdout.splitlines():
                if not rel or Path(rel).suffix not in SOURCE_EXTENSIONS:
                    continue
                if any(part in SKIP_DIR_NAMES for part in Path(rel).parts):
                    continue
                files.append(rel)
            if files:
                return files
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            logger.debug(f"git ls-files unavailable, falling back to walk: {exc}")

        files = []
        for path in repo_root.rglob("*"):
            if not path.is_file() or path.suffix not in SOURCE_EXTENSIONS:
                continue
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            files.append(str(path.relative_to(repo_root)))
            if len(files) >= self.config.max_scan_files:
                break
        return files

    def _prioritize_files(self, files: list[str], changed: set[str]) -> list[str]:
        changed_dirs = {str(Path(f).parent) for f in changed}

        def sort_key(rel: str) -> tuple[int, int, str]:
            parts = {p.lower() for p in Path(rel).parts}
            utility = 0 if parts & UTILITY_PATH_SEGMENTS else 1
            neighbor = 0 if str(Path(rel).parent) in changed_dirs else 1
            return (utility, neighbor, rel)

        return sorted(files, key=sort_key)

    def _search_candidates(
        self,
        repo_root: Path,
        candidate_files: list[str],
        anchors: list[str],
        changed_files: set[str],
    ) -> list[CapabilityHit]:
        hits: dict[str, CapabilityHit] = {}
        files_checked = 0

        for anchor in anchors:
            per_anchor = 0
            for rel in candidate_files:
                if rel in changed_files:
                    continue
                if files_checked >= self.config.max_scan_files:
                    break
                if per_anchor >= self.config.max_files_per_anchor:
                    break
                if rel in hits:
                    continue

                path = repo_root / rel
                if not path.is_file():
                    continue

                files_checked += 1
                match = self._find_anchor_line(path, anchor)
                if match is None:
                    continue

                lineno, signature = match
                score = 4.0 if self._is_utility_path(rel) else 2.0
                if str(Path(rel).stem).lower() == anchor:
                    score += 2.0
                if SYMBOL_LINE_RE.match(signature.strip()):
                    score += 1.5

                hits[rel] = CapabilityHit(
                    path=rel,
                    score=score,
                    reason=f"matches `{anchor}`",
                    signature=signature.strip(),
                    line=lineno,
                )
                per_anchor += 1

            if len(hits) >= self.config.max_snippets:
                break

        return sorted(hits.values(), key=lambda h: h.score, reverse=True)

    def _find_anchor_line(self, path: Path, anchor: str) -> tuple[int, str] | None:
        anchor_lower = anchor.lower()
        try:
            with path.open(encoding="utf-8", errors="ignore") as handle:
                for lineno, line in enumerate(handle, start=1):
                    lower = line.lower()
                    if anchor_lower not in lower:
                        continue
                    if SYMBOL_LINE_RE.match(line.strip()) or anchor_lower in lower:
                        return lineno, line.rstrip()
                    if lineno > 400:
                        break
        except OSError:
            return None
        return None

    def _apply_char_budget(
        self,
        hits: list[CapabilityHit],
        dependencies: list[str],
        anchors: list[str],
    ) -> list[CapabilityHit]:
        hits = hits[: self.config.max_snippets]
        kept: list[CapabilityHit] = []

        for hit in hits:
            trial = kept + [hit]
            if len(format_context_pack(trial, dependencies, anchors)) > self.config.max_chars:
                break
            kept = trial

        return kept

    def _relevant_dependencies(self, repo_root: Path, anchors: list[str]) -> list[str]:
        all_deps = self._parse_dependencies(repo_root)
        if not all_deps:
            return []

        anchor_set = set(anchors)
        relevant = [
            dep for dep in all_deps
            if any(a in dep.lower() for a in anchor_set)
        ]
        if relevant:
            return relevant[: self.config.max_deps]

        return all_deps[: self.config.max_deps]

    def _parse_dependencies(self, repo_root: Path) -> list[str]:
        deps: list[str] = []

        req = repo_root / "requirements.txt"
        if req.exists():
            for line in req.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    name = re.split(r"[<>=!~\[]", line)[0].strip()
                    if name:
                        deps.append(name)

        pyproject = repo_root / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib
            except ModuleNotFoundError:
                import toml as tomllib  # type: ignore[no-redef]

            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                for dep in data.get("project", {}).get("dependencies", []):
                    name = re.split(r"[<>=!~\[]", dep)[0].strip()
                    if name:
                        deps.append(name)
                for dep in data.get("tool", {}).get("poetry", {}).get("dependencies", {}):
                    deps.append(dep)
            except Exception:
                pass

        package_json = repo_root / "package.json"
        if package_json.exists():
            try:
                import json
                data = json.loads(package_json.read_text(encoding="utf-8"))
                for section in ("dependencies", "devDependencies"):
                    for name in data.get(section, {}):
                        deps.append(name)
            except Exception:
                pass

        return sorted(set(deps))

    def _is_utility_path(self, rel_path: str) -> bool:
        parts = {p.lower() for p in Path(rel_path).parts}
        return bool(parts & UTILITY_PATH_SEGMENTS)

    def _fingerprint(
        self,
        anchors: list[str],
        hits: list[CapabilityHit],
        dependencies: list[str],
    ) -> str:
        import hashlib

        payload = "|".join([
            ",".join(anchors),
            ",".join(h.path for h in hits),
            ",".join(dependencies),
        ])
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
