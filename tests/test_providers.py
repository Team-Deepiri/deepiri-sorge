"""Provider adapter mapping tests."""

from unittest.mock import MagicMock

import requests

from bot.diff_parser import ParsedDiff
from bot.file_splitter import ReviewChunk
from bot.providers._runner_adapter import run_runner_review
from bot.schemas import ReviewResult


def _chunk() -> ReviewChunk:
    return ReviewChunk(
        files=["a.py"],
        parsed_diff=ParsedDiff(raw="+x\n"),
        estimated_tokens=50,
    )


def test_adapter_maps_429():
    runner = MagicMock()
    resp = MagicMock()
    resp.status_code = 429
    resp.headers = {"Retry-After": "30"}
    err = requests.HTTPError(response=resp)
    err.response = resp
    runner.review.side_effect = err

    out = run_runner_review(provider_name="groq", runner=runner, chunk=_chunk(), run=MagicMock())
    assert out.ok is False
    assert out.status_code == 429
    assert out.retry_after == 30.0


def test_adapter_maps_413():
    runner = MagicMock()
    resp = MagicMock()
    resp.status_code = 413
    resp.headers = {}
    err = requests.HTTPError(response=resp)
    err.response = resp
    runner.review.side_effect = err

    out = run_runner_review(provider_name="gemini", runner=runner, chunk=_chunk(), run=MagicMock())
    assert out.is_payload_too_large


def test_adapter_maps_timeout():
    runner = MagicMock()
    runner.review.side_effect = requests.Timeout()
    out = run_runner_review(provider_name="openrouter", runner=runner, chunk=_chunk(), run=MagicMock())
    assert out.timed_out


def test_adapter_ok_result():
    runner = MagicMock()
    runner.review.return_value = ReviewResult(
        summary="ok",
        issues=[],
        recommendations=[],
        score=9.0,
        latency_ms=1,
        model="m",
        review_type="groq",
    )
    out = run_runner_review(provider_name="groq", runner=runner, chunk=_chunk(), run=MagicMock())
    assert out.ok
    assert out.result is not None


def test_adapter_surfaces_swallowed_http_status():
    """Runners that catch HTTPError and return None still expose _last_http_status."""
    runner = MagicMock()
    runner.review.return_value = None
    runner._last_http_status = 413
    runner._last_retry_after = None
    runner._last_timed_out = False
    out = run_runner_review(provider_name="groq", runner=runner, chunk=_chunk(), run=MagicMock())
    assert out.ok is False
    assert out.status_code == 413
    assert out.is_payload_too_large
