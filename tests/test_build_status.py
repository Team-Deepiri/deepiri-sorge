"""Tests for the CI build verdict used to disprove predicted compile failures."""

import requests

from bot import build_status
from bot.build_status import fetch_build_verdict


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def _patch(monkeypatch, payload=None, exc=None):
    def fake_get(url, headers=None, params=None, timeout=None):
        if exc is not None:
            raise exc
        return _FakeResponse(payload)

    monkeypatch.setattr(build_status.requests, "get", fake_get)


def _run(check_runs):
    return fetch_build_verdict(repo="o/r", sha="deadbeefcafe", token="t")


def test_all_checks_green_is_a_verdict(monkeypatch):
    _patch(
        monkeypatch,
        {
            "check_runs": [
                {"name": "rust", "status": "completed", "conclusion": "success"},
                {"name": "python", "status": "completed", "conclusion": "success"},
            ]
        },
    )
    green, reason = _run(None)
    assert green is True
    assert "rust" in reason


def test_a_failing_check_is_a_negative_verdict(monkeypatch):
    _patch(
        monkeypatch,
        {
            "check_runs": [
                {"name": "rust", "status": "completed", "conclusion": "failure"},
                {"name": "python", "status": "completed", "conclusion": "success"},
            ]
        },
    )
    green, reason = _run(None)
    assert green is False
    assert "rust" in reason


def test_checks_still_running_yield_no_verdict(monkeypatch):
    """A review can fire before CI finishes; that must not suppress anything."""
    _patch(
        monkeypatch,
        {
            "check_runs": [
                {"name": "rust", "status": "in_progress", "conclusion": None},
                {"name": "python", "status": "completed", "conclusion": "success"},
            ]
        },
    )
    green, reason = _run(None)
    assert green is None
    assert "still running" in reason


def test_no_checks_at_all_yields_no_verdict(monkeypatch):
    _patch(monkeypatch, {"check_runs": []})
    green, _ = _run(None)
    assert green is None


def test_non_build_checks_alone_yield_no_verdict(monkeypatch):
    """A green CodeQL run says nothing about whether the code compiles."""
    _patch(
        monkeypatch,
        {
            "check_runs": [
                {"name": "CodeQL", "status": "completed", "conclusion": "success"},
                {"name": "codecov/patch", "status": "completed", "conclusion": "success"},
            ]
        },
    )
    green, reason = _run(None)
    assert green is None
    assert "build-relevant" in reason


def test_failing_non_build_check_does_not_veto_a_green_build(monkeypatch):
    """A red linter must not read as a broken build."""
    _patch(
        monkeypatch,
        {
            "check_runs": [
                {"name": "rust", "status": "completed", "conclusion": "success"},
                {"name": "CodeQL", "status": "completed", "conclusion": "failure"},
            ]
        },
    )
    green, _ = _run(None)
    assert green is True


def test_skipped_only_is_not_evidence_of_success(monkeypatch):
    _patch(
        monkeypatch,
        {"check_runs": [{"name": "rust", "status": "completed", "conclusion": "skipped"}]},
    )
    green, reason = _run(None)
    assert green is None
    assert "success" in reason


def test_network_failure_yields_no_verdict(monkeypatch):
    _patch(monkeypatch, exc=requests.ConnectionError("boom"))
    green, reason = _run(None)
    assert green is None
    assert "fetch failed" in reason


def test_missing_inputs_short_circuit_without_a_request(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("must not call the API without repo/sha/token")

    monkeypatch.setattr(build_status.requests, "get", explode)

    assert fetch_build_verdict(repo="", sha="abc", token="t")[0] is None
    assert fetch_build_verdict(repo="o/r", sha="", token="t")[0] is None
    assert fetch_build_verdict(repo="o/r", sha="abc", token="")[0] is None
