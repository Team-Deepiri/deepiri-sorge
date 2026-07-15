"""Tests for main entrypoint runner dispatch."""

import json
from argparse import Namespace

from bot.runners.base import ReviewResult
from bot.decision_engine import Action, ReviewDecision
from bot.diff_parser import ParsedDiff
from bot.main import main, parse_args


def _sample_result(review_type: str) -> ReviewResult:
    return ReviewResult(
        summary="ok",
        issues=[],
        recommendations=[],
        score=9.0,
        model="test-model",
        latency_ms=0.0,
        review_type=review_type,
    )


def _sample_diff() -> ParsedDiff:
    return ParsedDiff(
        raw="diff --git a/a.py b/a.py\n+print('x')",
        files=["a.py"],
        file_changes={},
        lines_added=1,
        lines_deleted=0,
        files_changed=1,
    )


def test_parse_args_accepts_openrouter_and_groq(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--mode", "openrouter", "--diff", "diff --git a/a.py b/a.py"],
    )
    args = parse_args()
    assert args.mode == "openrouter"

    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--mode", "groq", "--diff", "diff --git a/a.py b/a.py"],
    )
    args = parse_args()
    assert args.mode == "groq"


def test_main_dispatches_openrouter_for_explicit_mode(monkeypatch, capsys):
    called = {"openrouter": False}

    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    class FakeOpenRouterRunner:
        def __init__(self, *args, **kwargs):
            self.model = kwargs.get("model") or "fake"
            self._last_http_status = None
            self._last_retry_after = None
            self._last_timed_out = False

        def review(self, parsed_diff, **kwargs):
            called["openrouter"] = True
            return _sample_result("openrouter")

    monkeypatch.setattr(
        "bot.main.parse_args",
        lambda: Namespace(
            diff="inline-diff",
            config="does-not-exist.toml",
            pr_number=None,
            repo=None,
            token=None,
            installation_id=None,
            dry_run=True,
            verbose=False,
            mode="openrouter",
            repo_root=".",
            force=False,
        ),
    )
    monkeypatch.setattr(
        "bot.main.RepoContextWeaver",
        lambda config: type(
            "W",
            (),
            {"weave": lambda self, root, diff: type("P", (), {"text": "", "fingerprint": ""})()},
        )(),
    )
    monkeypatch.setattr("bot.main.load_diff", lambda _: "diff-content")
    monkeypatch.setattr("bot.providers.openrouter.OpenRouterRunner", FakeOpenRouterRunner)
    monkeypatch.setattr("bot.main.DiffParser.parse", lambda self, _: _sample_diff())
    monkeypatch.setattr(
        "bot.main.DecisionEngine.decide",
        lambda self, _: ReviewDecision(action=Action.OPENROUTER, reason="test"),
    )

    main()

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert called["openrouter"] is True
    assert payload["review_type"] in ("openrouter", "aggregated")


def test_main_dispatches_groq_for_auto_mode_decision(monkeypatch, capsys):
    called = {"auto": False}

    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    def fake_auto(*args, **kwargs):
        called["auto"] = True
        return _sample_result("groq")

    monkeypatch.setattr(
        "bot.main.parse_args",
        lambda: Namespace(
            diff="inline-diff",
            config="does-not-exist.toml",
            pr_number=None,
            repo=None,
            token=None,
            installation_id=None,
            dry_run=True,
            verbose=False,
            mode="auto",
            repo_root=".",
            force=False,
        ),
    )
    monkeypatch.setattr(
        "bot.main.RepoContextWeaver",
        lambda config: type(
            "W",
            (),
            {"weave": lambda self, root, diff: type("P", (), {"text": "", "fingerprint": ""})()},
        )(),
    )
    monkeypatch.setattr("bot.main.load_diff", lambda _: "diff-content")
    monkeypatch.setattr("bot.main.run_auto_review", fake_auto)
    monkeypatch.setattr("bot.main.DiffParser.parse", lambda self, _: _sample_diff())
    monkeypatch.setattr(
        "bot.main.DecisionEngine.decide",
        lambda self, _: ReviewDecision(action=Action.GROQ, reason="test"),
    )

    main()

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert called["auto"] is True
    assert payload["review_type"] == "groq"
