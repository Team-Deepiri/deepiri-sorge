"""PR-HEAD dependency evidence for the review prompt.

Gives the model objective facts about packages imported in the DIFF vs
manifests on the checked-out tip — so it does not guess from hunks alone.
ClaimVerifier remains the post-condition safety net for the same class of claims.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger

from bot.diff_parser import ParsedDiff

# Added-line import extractors (JS/TS/Python).
_JS_FROM_RE = re.compile(
    r"""^\+\s*import\s+(?:type\s+)?[\w*{}\s,]*\s+from\s+['"]([^'"]+)['"]"""
)
_JS_SIDE_RE = re.compile(r"""^\+\s*import\s+['"]([^'"]+)['"]""")
_JS_REQUIRE_RE = re.compile(
    r"""^\+\s*(?:const|let|var)\s+\w+\s*=\s*require\(\s*['"]([^'"]+)['"]\s*\)"""
)
_JS_ISH_RE = re.compile(
    r"""^\+\s*import\s+(?:type\b|[\w*{].*\s+from\s+['"]|['"])"""
)
_PY_IMPORT_RE = re.compile(
    r"^\+\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))(?:\s*$|\s*[;#])"
)

# Node core modules (bare + node: protocol). Keep compact — prefix rules cover most.
_NODE_BUILTINS = frozenset({
    "assert", "async_hooks", "buffer", "child_process", "cluster", "console",
    "constants", "crypto", "dgram", "diagnostics_channel", "dns", "domain",
    "events", "fs", "http", "http2", "https", "inspector", "module", "net",
    "os", "path", "perf_hooks", "process", "punycode", "querystring",
    "readline", "repl", "stream", "string_decoder", "sys", "timers", "tls",
    "trace_events", "tty", "url", "util", "v8", "vm", "wasi", "worker_threads",
    "zlib", "test",
})


def load_declared_dependencies(repo_root: Path) -> set[str]:
    """Package names declared on PR HEAD (npm + common Python manifests)."""
    names: set[str] = set()
    root = Path(repo_root)

    pkg = root / "package.json"
    if pkg.is_file():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            for section in (
                "dependencies",
                "devDependencies",
                "optionalDependencies",
                "peerDependencies",
            ):
                block = data.get(section) or {}
                if isinstance(block, dict):
                    names.update(str(k) for k in block.keys())

    req = root / "requirements.txt"
    if req.is_file():
        try:
            for line in req.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                token = re.split(r"[<>=!~;\s\[]", line, maxsplit=1)[0].strip()
                if token:
                    names.add(token)
        except OSError:
            pass

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            text = pyproject.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if text:
            try:
                import tomllib
            except ModuleNotFoundError:
                import toml as tomllib  # type: ignore[no-redef]
            try:
                data = tomllib.loads(text)
                for dep in data.get("project", {}).get("dependencies", []) or []:
                    name = re.split(r"[<>=!~\[]", str(dep), maxsplit=1)[0].strip()
                    if name:
                        names.add(name)
                poetry = (
                    data.get("tool", {}).get("poetry", {}).get("dependencies", {}) or {}
                )
                if isinstance(poetry, dict):
                    for name in poetry:
                        if name != "python":
                            names.add(str(name))
            except Exception:
                pass

    return names


def extract_imported_packages(diff: ParsedDiff) -> list[str]:
    """Third-party package names newly imported in added diff lines."""
    found: list[str] = []
    seen: set[str] = set()

    for line in (diff.raw or "").splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        candidates: list[str] = []
        js_hit = False
        for pattern in (_JS_FROM_RE, _JS_SIDE_RE, _JS_REQUIRE_RE):
            m = pattern.match(line)
            if m:
                candidates.append(m.group(1))
                js_hit = True
        # Don't let Python `import X` match JS/TS (`import type`, `import X from "..."`).
        if not js_hit and not _JS_ISH_RE.match(line):
            m = _PY_IMPORT_RE.match(line)
            if m:
                candidates.append(m.group(1) or m.group(2) or "")

        for raw in candidates:
            pkg = _normalize_import_package(raw)
            if not pkg or pkg in seen:
                continue
            seen.add(pkg)
            found.append(pkg)

    return found


def _normalize_import_package(raw: str) -> str | None:
    name = (raw or "").strip()
    if not name or name.startswith(".") or name.startswith("/"):
        return None
    # node: protocol builtins (node:fs, node:test, …)
    if name.startswith("node:"):
        return None
    if name.startswith("@"):
        parts = name.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return name
    # package subpath: lodash/get → lodash; pdf-parse stays
    root = name.split("/")[0]
    if root in _NODE_BUILTINS:
        return None
    return root


def format_import_manifest_evidence(
    repo_root: Path,
    diff: ParsedDiff,
    *,
    max_chars: int = 1200,
    max_packages: int = 40,
) -> str:
    """Compact IMPORT_VS_MANIFEST block for REPOSITORY CONTEXT."""
    packages = extract_imported_packages(diff)[:max_packages]
    if not packages:
        logger.info("IMPORT_VS_MANIFEST: skipped (no third-party imports in DIFF)")
        return ""

    declared = load_declared_dependencies(repo_root)
    has_manifest = (
        (Path(repo_root) / "package.json").is_file()
        or (Path(repo_root) / "requirements.txt").is_file()
        or (Path(repo_root) / "pyproject.toml").is_file()
    )
    if not declared and not has_manifest:
        logger.info("IMPORT_VS_MANIFEST: skipped (no readable HEAD manifests)")
        return ""

    declared_l = {n.lower(): n for n in declared}
    lines = [
        "IMPORT_VS_MANIFEST (PR HEAD) — objective dependency facts for packages "
        "imported in the DIFF. Do not claim a package is missing from package.json / "
        "requirements / pyproject when listed as declared here:",
    ]
    declared_n = 0
    missing_n = 0
    for pkg in packages:
        key = pkg.lower()
        # Python import foo_bar vs package foo-bar
        alt = pkg.replace("_", "-").lower()
        if key in declared_l or alt in declared_l or pkg in declared:
            canon = declared_l.get(key) or declared_l.get(alt) or pkg
            lines.append(f"  {pkg}: declared ({canon})")
            declared_n += 1
        else:
            lines.append(f"  {pkg}: NOT declared on HEAD manifests")
            missing_n += 1

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n  …(truncated)"

    logger.info(
        f"IMPORT_VS_MANIFEST: injected {len(packages)} import(s) "
        f"({declared_n} declared, {missing_n} missing) chars={len(text)}"
    )
    return text
