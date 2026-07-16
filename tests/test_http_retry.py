"""HTTP retry policy for Phase 1 scheduler."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from tests.helpers import install_loguru_stub

install_loguru_stub()

from bot.utils.http_retry import post_with_retry


def test_retries_on_429_once_then_succeeds():
    ok = MagicMock()
    ok.status_code = 200
    ok.raise_for_status = MagicMock()
    ok.headers = {}

    rate_limited = MagicMock()
    rate_limited.status_code = 429
    rate_limited.headers = {"Retry-After": "2"}
    rate_limited.raise_for_status = MagicMock()

    with patch("bot.utils.http_retry.requests.post", side_effect=[rate_limited, ok]) as post:
        with patch("bot.utils.http_retry.time.sleep") as sleep:
            response = post_with_retry("https://example.com", json={"a": 1})

    assert response is ok
    assert post.call_count == 2
    sleep.assert_called()
    assert sleep.call_args[0][0] == 2.0


def test_does_not_retry_413():
    too_large = MagicMock()
    too_large.status_code = 413
    too_large.headers = {}
    too_large.raise_for_status.side_effect = requests.HTTPError(
        "413", response=too_large
    )

    with patch("bot.utils.http_retry.requests.post", return_value=too_large) as post:
        with patch("bot.utils.http_retry.time.sleep") as sleep:
            with pytest.raises(requests.HTTPError):
                post_with_retry("https://example.com", json={}, max_retries=3)

    assert post.call_count == 1
    sleep.assert_not_called()


def test_raises_after_max_retries_on_503():
    rate_limited = MagicMock()
    rate_limited.status_code = 503
    rate_limited.headers = {}
    rate_limited.raise_for_status.side_effect = requests.HTTPError("503")

    with patch("bot.utils.http_retry.requests.post", return_value=rate_limited):
        with patch("bot.utils.http_retry.time.sleep"):
            with pytest.raises(requests.HTTPError):
                post_with_retry("https://example.com", json={}, max_retries=2)
