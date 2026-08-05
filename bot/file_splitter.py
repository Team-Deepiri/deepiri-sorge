"""Dependency-aware file splitting for oversized PRs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from bot.diff_parser import ParsedDiff, estimate_tokens


# Language-specific import patterns keyed by file extension.
_IMPORT_PATTERNS: dict[str, list[re.Pattern]] = {
    ".py": [
        re.compile(r"^from\s+([\w.]+)"),
        re.compile(r"^import\s+([\w.]+)"),
    ],
    ".js": [
        re.compile(r"""import\s+['"]([^'"]+)['"]"""),
        re.compile(r"""from\s+['"]([^'"]+)['"]"""),
    ],
    ".jsx": [
        re.compile(r"""import\s+['"]([^'"]+)['"]"""),
        re.compile(r"""from\s+['"]([^'"]+)['"]"""),
    ],
    ".ts": [
        re.compile(r"""import\s+['"]([^'"]+)['"]"""),
        re.compile(r"""from\s+['"]([^'"]+)['"]"""),
    ],
    ".tsx": [
        re.compile(r"""import\s+['"]([^'"]+)['"]"""),
        re.compile(r"""from\s+['"]([^'"]+)['"]"""),
    ],
    ".java": [
        re.compile(r"^import\s+([\w.*]+)\s*;"),
    ],
    ".rs": [
        re.compile(r"^use\s+([\w:]+)"),
    ],
    ".go": [
        re.compile(r"""import\s+['"]([^'"]+)['"]"""),
    ],
    ".cs": [
        re.compile(r"^using\s+([\w.]+)"),
    ],
    ".c": [
        re.compile(r"""#include\s+[<"]([^'">]+)[>"]"""),
    ],
    ".cpp": [
        re.compile(r"""#include\s+[<"]([^'">]+)[>"]"""),
    ],
    ".cc": [
        re.compile(r"""#include\s+[<"]([^'">]+)[>"]"""),
    ],
    ".h": [
        re.compile(r"""#include\s+[<"]([^'">]+)[>"]"""),
    ],
    ".hh": [
        re.compile(r"""#include\s+[<"]([^'">]+)[>"]"""),
    ],
    ".hpp": [
        re.compile(r"""#include\s+[<"]([^'">]+)[>"]"""),
    ],
}

_SOURCE_EXTS = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".rs", ".go", ".cs",
    ".c", ".cpp", ".cc", ".h", ".hh", ".hpp",
})


@dataclass
class ReviewChunk:
    files: list[str]
    parsed_diff: ParsedDiff
    estimated_tokens: int
    part_label: str | None = None
    unreviewable: bool = False
    unreviewable_reason: str | None = None


@dataclass
class FileSplitter:
    chunk_budget: int = 180_000
    max_chunk_tokens: int = 200_000

    def split(self, diff: ParsedDiff) -> list[ReviewChunk]:
        total = estimate_tokens(diff.raw)
        if total <= self.chunk_budget:
            return [
                ReviewChunk(
                    files=list(diff.files),
                    parsed_diff=diff,
                    estimated_tokens=total,
                )
            ]

        groups = self._dependency_groups(diff)
        chunks: list[ReviewChunk] = []

        # Dependency grouping keeps import-connected files together for
        # context, but a PR spanning several unrelated packages (or lots of
        # config/CSS/docs with no detected import edges) leaves most files
        # as their own singleton group. Packing each group separately then
        # burns one full LLM request per file even when a dozen of them
        # would comfortably fit in a single chunk together. Split off the
        # groups that already fit within budget on their own and bin-pack
        # those together; only groups that need real splitting go through
        # `_pack_group` individually.
        small_groups: list[list[str]] = []
        for group in groups:
            if self._group_tokens(diff, group) <= self.chunk_budget:
                small_groups.append(group)
            else:
                chunks.extend(self._pack_group(diff, group))

        chunks.extend(self._bin_pack_groups(diff, small_groups))

        return chunks or [
            ReviewChunk(
                files=list(diff.files),
                parsed_diff=diff,
                estimated_tokens=estimate_tokens(diff.raw),
            )
        ]

    def _group_tokens(self, diff: ParsedDiff, group: list[str]) -> int:
        return sum(
            estimate_tokens(diff.file_changes[p].raw_diff)
            for p in group
            if p in diff.file_changes
        )

    def _bin_pack_groups(
        self, diff: ParsedDiff, groups: list[list[str]]
    ) -> list[ReviewChunk]:
        """Greedily combine unrelated small groups into shared chunks.

        Each input group already fits within ``chunk_budget`` on its own;
        this just avoids sending N nearly-empty requests when they'd fit
        together in far fewer. Largest-first bin packing keeps the chunk
        count low.
        """
        if not groups:
            return []

        sized = sorted(
            ((g, self._group_tokens(diff, g)) for g in groups),
            key=lambda gt: gt[1],
            reverse=True,
        )

        chunks: list[ReviewChunk] = []
        current_files: list[str] = []
        current_tokens = 0

        for group, tokens in sized:
            if current_files and current_tokens + tokens > self.chunk_budget:
                sub = diff.slice_files(current_files)
                chunks.append(
                    ReviewChunk(
                        files=current_files,
                        parsed_diff=sub,
                        estimated_tokens=estimate_tokens(sub.raw),
                    )
                )
                current_files = []
                current_tokens = 0
            current_files.extend(group)
            current_tokens += tokens

        if current_files:
            sub = diff.slice_files(current_files)
            chunks.append(
                ReviewChunk(
                    files=current_files,
                    parsed_diff=sub,
                    estimated_tokens=estimate_tokens(sub.raw),
                )
            )

        return chunks

    def _patterns_for(self, path: str) -> list[re.Pattern]:
        """Return import patterns matching the file extension."""
        ext = Path(path).suffix
        return _IMPORT_PATTERNS.get(ext, [])

    def _normalise_ref(self, ref: str) -> str:
        """Normalise an import reference to dotted module form."""
        ref = ref.replace("/", ".").replace("::", ".")
        for ext in _SOURCE_EXTS:
            ref = ref.removesuffix(ext)
        return ref

    def _dependency_groups(self, diff: ParsedDiff) -> list[list[str]]:
        paths = list(diff.files)
        if not paths:
            return []

        path_set = set(paths)
        module_to_path: dict[str, str] = {}
        for p in paths:
            mod = self._path_to_module(p)
            module_to_path[mod] = p
            base = mod.split(".")[0]
            if base not in module_to_path:
                module_to_path[base] = p

        adjacency: dict[str, set[str]] = {p: set() for p in paths}

        for path in paths:
            patterns = self._patterns_for(path)
            if not patterns:
                continue
            change = diff.file_changes.get(path)
            if not change or not change.raw_diff:
                continue
            for line in change.raw_diff.split("\n"):
                if not (line.startswith("+") or line.startswith("-")):
                    continue
                content = line[1:].strip()
                for pattern in patterns:
                    m = pattern.match(content)
                    if not m:
                        continue
                    ref = self._normalise_ref(m.group(1))
                    target = module_to_path.get(ref) or module_to_path.get(ref.split(".")[0])
                    if target and target in path_set and target != path:
                        adjacency[path].add(target)
                        adjacency[target].add(path)

        visited: set[str] = set()
        groups: list[list[str]] = []

        for path in paths:
            if path in visited:
                continue
            stack = [path]
            component: list[str] = []
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                stack.extend(adjacency.get(node, set()) - visited)
            groups.append(sorted(component))

        return groups

    def _pack_group(self, diff: ParsedDiff, group: list[str]) -> list[ReviewChunk]:
        group_tokens = sum(
            estimate_tokens(diff.file_changes[p].raw_diff)
            for p in group
            if p in diff.file_changes
        )

        if group_tokens <= self.chunk_budget:
            sub = diff.slice_files(group)
            return [
                ReviewChunk(
                    files=group,
                    parsed_diff=sub,
                    estimated_tokens=estimate_tokens(sub.raw),
                )
            ]

        chunks: list[ReviewChunk] = []
        current_files: list[str] = []
        current_tokens = 0

        for path in sorted(group, key=lambda p: estimate_tokens(diff.file_changes[p].raw_diff), reverse=True):
            ft = estimate_tokens(diff.file_changes[path].raw_diff) if path in diff.file_changes else 0

            if ft > self.max_chunk_tokens:
                if current_files:
                    sub = diff.slice_files(current_files)
                    chunks.append(
                        ReviewChunk(
                            files=current_files,
                            parsed_diff=sub,
                            estimated_tokens=estimate_tokens(sub.raw),
                        )
                    )
                    current_files = []
                    current_tokens = 0
                chunks.extend(self._split_large_file(diff, path))
                continue

            if current_tokens + ft > self.chunk_budget and current_files:
                sub = diff.slice_files(current_files)
                chunks.append(
                    ReviewChunk(
                        files=current_files,
                        parsed_diff=sub,
                        estimated_tokens=estimate_tokens(sub.raw),
                    )
                )
                current_files = []
                current_tokens = 0

            current_files.append(path)
            current_tokens += ft

        if current_files:
            sub = diff.slice_files(current_files)
            chunks.append(
                ReviewChunk(
                    files=current_files,
                    parsed_diff=sub,
                    estimated_tokens=estimate_tokens(sub.raw),
                )
            )

        return chunks

    def _split_large_file(self, diff: ParsedDiff, path: str) -> list[ReviewChunk]:
        change = diff.file_changes.get(path)
        if not change:
            return []

        n_hunks = len(change.hunks) or 1
        chunks: list[ReviewChunk] = []
        batch: list[int] = []
        batch_tokens = estimate_tokens(change.raw_diff[:0])

        for i in range(n_hunks):
            trial = diff.slice_hunks(path, batch + [i])
            trial_tokens = estimate_tokens(trial.raw)

            if trial_tokens > self.max_chunk_tokens and batch:
                sub = diff.slice_hunks(path, batch)
                chunks.append(
                    ReviewChunk(
                        files=[path],
                        parsed_diff=sub,
                        estimated_tokens=estimate_tokens(sub.raw),
                        part_label=f"{path} (part {len(chunks) + 1})",
                    )
                )
                batch = [i]
                batch_tokens = estimate_tokens(diff.slice_hunks(path, batch).raw)
            elif trial_tokens > self.max_chunk_tokens:
                chunks.append(
                    ReviewChunk(
                        files=[path],
                        parsed_diff=trial,
                        estimated_tokens=trial_tokens,
                        unreviewable=True,
                        unreviewable_reason=(
                            f"{path}: hunk batch exceeds {self.max_chunk_tokens} token limit"
                        ),
                    )
                )
                batch = []
                batch_tokens = 0
            else:
                batch.append(i)
                batch_tokens = trial_tokens

        if batch:
            sub = diff.slice_hunks(path, batch)
            chunks.append(
                ReviewChunk(
                    files=[path],
                    parsed_diff=sub,
                    estimated_tokens=estimate_tokens(sub.raw),
                    part_label=f"{path} (part {len(chunks) + 1})",
                )
            )

        return chunks

    def _path_to_module(self, path: str) -> str:
        p = path.replace("/", ".").replace("\\", ".")
        for ext in _SOURCE_EXTS:
            p = p.removesuffix(ext)
        if p.endswith(".__init__"):
            p = p[: -len(".__init__")]
        return p
