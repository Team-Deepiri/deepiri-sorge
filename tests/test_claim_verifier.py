"""Tests for symbol index + conservative claim verifier."""

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
