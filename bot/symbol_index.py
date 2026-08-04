"""Lightweight post-image symbol maps for changed source files.

Builds a compact, ordered list of module-level bindings (classes, functions,
simple aliases, imports) from the PR-head / post-change file contents.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

PYTHON_EXTENSIONS = frozenset({".py"})
CPP_EXTENSIONS = frozenset({
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx",
})
SOURCE_EXTENSIONS = PYTHON_EXTENSIONS | CPP_EXTENSIONS

# Keywords that take a parenthesised clause and would otherwise read as calls.
_CPP_CONTROL_KEYWORDS = frozenset({
    "if", "else", "for", "while", "switch", "catch", "return", "sizeof",
    "do", "case", "throw", "new", "delete", "static_assert", "decltype",
    "noexcept", "alignof", "and", "or", "not", "constexpr", "explicit",
})

_CPP_INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]")
_CPP_DEFINE_RE = re.compile(r"^\s*#\s*define\s+(\w+)")
_CPP_TYPE_RE = re.compile(
    r"^\s*(?:template\s*<[^>]*>\s*)?"
    r"(class|struct|union|namespace|enum(?:\s+class)?)\s+(\w+)"
)
_CPP_USING_RE = re.compile(r"^\s*using\s+(\w+)\s*=")
_CPP_TYPEDEF_RE = re.compile(r"^\s*typedef\s+.+?\b(\w+)\s*;")
# Definition (ends in `{`) or declaration (ends in `;`) with a parameter list.
_CPP_FUNC_RE = re.compile(
    r"^\s*(?:template\s*<[^>]*>\s*)?"
    r"(?:[A-Za-z_][\w:<>,\s\*&\[\]]*?\s[\*&\s]*)?"
    r"(~?\w+)\s*\([^;{]*\)\s*"
    r"(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:final\s*)?"
    r"(?:=\s*(?:0|default|delete)\s*)?"
    r"[;{]"
)


@dataclass(frozen=True)
class SymbolBinding:
    name: str
    line: int
    kind: str  # class | function | alias | import
    target: str | None = None


@dataclass
class FileSymbolIndex:
    path: str
    bindings: list[SymbolBinding] = field(default_factory=list)

    def first_binding(self, name: str) -> SymbolBinding | None:
        for binding in self.bindings:
            if binding.name == name:
                return binding
        return None

    def names(self) -> set[str]:
        return {b.name for b in self.bindings}


def format_symbol_index(indexes: list[FileSymbolIndex], *, max_chars: int = 2400) -> str:
    """Human-readable compact symbol index for optional prompt injection."""
    if not indexes:
        return ""

    parts: list[str] = [
        "SYMBOL_INDEX (post-change / PR head) — use for symbol resolution; "
        "do not claim NameError / undefined / used-before-defined unless this "
        "index clearly shows the symbol is missing or bound after the use site:",
    ]

    for index in indexes:
        block: list[str] = [f"\n{index.path}"]
        classes = [b for b in index.bindings if b.kind == "class"]
        funcs = [b for b in index.bindings if b.kind == "function"]
        aliases = [b for b in index.bindings if b.kind == "alias"]
        imports = [b for b in index.bindings if b.kind == "import"]

        if classes:
            block.append("  Classes")
            block.extend(f"    {b.name} @{b.line}" for b in classes)
        if aliases:
            block.append("  Aliases")
            for b in aliases:
                target = f" -> {b.target}" if b.target else ""
                block.append(f"    {b.name}{target} @{b.line}")
        if funcs:
            block.append("  Functions")
            block.extend(f"    {b.name} @{b.line}" for b in funcs)
        if imports:
            block.append("  Imports")
            for b in imports[:12]:
                target = f" ({b.target})" if b.target else ""
                block.append(f"    {b.name}{target} @{b.line}")

        trial = "\n".join(parts + block)
        if len(trial) > max_chars and len(parts) > 1:
            break
        parts.extend(block)

    return "\n".join(parts)[:max_chars]


class SymbolIndexer:
    """Parse changed files under repo_root into FileSymbolIndex objects."""

    def index_files(self, repo_root: Path, paths: list[str]) -> list[FileSymbolIndex]:
        repo_root = repo_root.resolve()
        indexes: list[FileSymbolIndex] = []
        for rel in paths:
            if Path(rel).suffix not in SOURCE_EXTENSIONS:
                continue
            path = repo_root / rel
            if not path.is_file():
                logger.debug(f"Symbol index skip (missing): {rel}")
                continue
            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                logger.debug(f"Symbol index read failed for {rel}: {exc}")
                continue
            if Path(rel).suffix in CPP_EXTENSIONS:
                bindings = self._parse_cpp(source)
            else:
                bindings = self._parse_python(source)
            if bindings:
                indexes.append(FileSymbolIndex(path=rel, bindings=bindings))
        return indexes

    def _parse_cpp(self, source: str) -> list[SymbolBinding]:
        """Regex scan for C/C++ bindings.

        Unlike the Python path this indexes at any nesting depth: C++ puts
        member functions inside class bodies and splits declaration from
        definition across header and source, so a module-scope-only index
        would be empty for exactly the files we care about.
        """
        bindings: list[SymbolBinding] = []
        seen: set[tuple[str, int]] = set()
        in_block_comment = False

        def add(name: str, lineno: int, kind: str, target: str | None = None) -> None:
            if not name or (name, lineno) in seen:
                return
            seen.add((name, lineno))
            bindings.append(SymbolBinding(name, lineno, kind, target=target))

        for lineno, raw in enumerate(source.splitlines(), start=1):
            line = raw
            if in_block_comment:
                end = line.find("*/")
                if end == -1:
                    continue
                line = line[end + 2 :]
                in_block_comment = False
            start = line.find("/*")
            if start != -1:
                if "*/" in line[start:]:
                    line = line[:start] + line[line.find("*/", start) + 2 :]
                else:
                    line = line[:start]
                    in_block_comment = True
            line = line.split("//", 1)[0]
            if not line.strip():
                continue

            if match := _CPP_INCLUDE_RE.match(line):
                header = match.group(1)
                add(Path(header).stem, lineno, "import", target=header)
                continue
            if match := _CPP_DEFINE_RE.match(line):
                add(match.group(1), lineno, "alias")
                continue
            if match := _CPP_TYPE_RE.match(line):
                kind = "class" if not match.group(1).startswith("namespace") else "alias"
                add(match.group(2), lineno, kind)
                continue
            if match := _CPP_USING_RE.match(line):
                add(match.group(1), lineno, "alias")
                continue
            if match := _CPP_TYPEDEF_RE.match(line):
                add(match.group(1), lineno, "alias")
                continue
            if match := _CPP_FUNC_RE.match(line):
                name = match.group(1)
                if name.lstrip("~") in _CPP_CONTROL_KEYWORDS:
                    continue
                add(name, lineno, "function")

        return bindings

    def _parse_python(self, source: str) -> list[SymbolBinding]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._fallback_scan(source)

        bindings: list[SymbolBinding] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bindings.append(SymbolBinding(node.name, node.lineno, "class"))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bindings.append(SymbolBinding(node.name, node.lineno, "function"))
            elif isinstance(node, ast.Assign):
                target_name = self._simple_name(node.value)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bindings.append(
                            SymbolBinding(
                                target.id,
                                node.lineno,
                                "alias",
                                target=target_name,
                            )
                        )
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                bindings.append(
                    SymbolBinding(
                        node.target.id,
                        node.lineno,
                        "alias",
                        target=self._simple_name(node.value) if node.value else None,
                    )
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    bindings.append(
                        SymbolBinding(name, node.lineno, "import", target=alias.name)
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    name = alias.asname or alias.name
                    bindings.append(
                        SymbolBinding(
                            name,
                            node.lineno,
                            "import",
                            target=f"{module}.{alias.name}" if module else alias.name,
                        )
                    )
        return bindings

    def _simple_name(self, node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            cur: ast.AST | None = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                return ".".join(reversed(parts))
        return None

    def _fallback_scan(self, source: str) -> list[SymbolBinding]:
        """Line-oriented fallback when AST parse fails."""
        bindings: list[SymbolBinding] = []
        class_re = re.compile(r"^class\s+(\w+)")
        def_re = re.compile(r"^(?:async\s+)?def\s+(\w+)")
        alias_re = re.compile(r"^([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*$")
        for lineno, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if m := class_re.match(stripped):
                bindings.append(SymbolBinding(m.group(1), lineno, "class"))
            elif m := def_re.match(stripped):
                bindings.append(SymbolBinding(m.group(1), lineno, "function"))
            elif m := alias_re.match(stripped):
                bindings.append(SymbolBinding(m.group(1), lineno, "alias", target=m.group(2)))
        return bindings
