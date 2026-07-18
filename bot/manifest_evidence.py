"""PR-HEAD dependency evidence for the review prompt.

Gives the model objective facts about packages imported in the DIFF vs
manifests on the checked-out tip — so it does not guess from hunks alone.
ClaimVerifier remains the post-condition safety net for the same class of claims.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from bot.diff_parser import ParsedDiff

# Added-line import extractors (JS/TS/Python).
_JS_FROM_RE = re.compile(
    r"""^\+\s*import\s+(?:type\s+)?[\w*{}\s,]*\s+from\s+['"]([^'"]+)['"]"""
)
_JS_SIDE_RE = re.compile(r"""^\+\s*import\s+['"]([^'"]+)['"]""")
_JS_REQUIRE_RE = re.compile(
    r"""^\+\s*(?:const|let|var)\s+\w+\s*=\s*require\(\s*['"]([^'"]+)['"]\s*\)"""
)
_PY_IMPORT_RE = re.compile(
    r"^\+\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))"
)


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
        if not js_hit:
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
    # Node builtins / common relative-ish noise
    if name in {"fs", "path", "os", "util", "http", "https", "crypto", "url"}:
        return None
    if name.startswith("@"):
        parts = name.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return name
    # package subpath: lodash/get → lodash; pdf-parse stays
    return name.split("/")[0]


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
        return ""

    declared = load_declared_dependencies(repo_root)
    if not declared and not (Path(repo_root) / "package.json").is_file():
        # No readable npm/python manifests — skip (don't invent evidence).
        req = Path(repo_root) / "requirements.txt"
        pyproject = Path(repo_root) / "pyproject.toml"
        if not req.is_file() and not pyproject.is_file():
            return ""

    declared_l = {n.lower(): n for n in declared}
    lines = [
        "IMPORT_VS_MANIFEST (PR HEAD) — objective dependency facts for packages "
        "imported in the DIFF. Do not claim a package is missing from package.json / "
        "requirements / pyproject when listed as declared here:",
    ]
    for pkg in packages:
        key = pkg.lower()
        # Python import foo_bar vs package foo-bar
        alt = pkg.replace("_", "-").lower()
        if key in declared_l or alt in declared_l or pkg in declared:
            canon = declared_l.get(key) or declared_l.get(alt) or pkg
            lines.append(f"  {pkg}: declared ({canon})")
        else:
            lines.append(f"  {pkg}: NOT declared on HEAD manifests")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n  …(truncated)"
    return text
