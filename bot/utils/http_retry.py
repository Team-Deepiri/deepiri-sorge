"""HTTP retry with bounded 429 handling and non-retryable 413."""

from __future__ import annotations

import time
from typing import Any

import requests
from loguru import logger

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
NEVER_RETRY_STATUS = frozenset({400, 401, 403, 404, 413, 422})
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SEC = 1.5


def _retry_after_seconds(response: requests.Response | None) -> float | None:
    if response is None:
        return None
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def post_with_retry(
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: int = 120,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_sec: float = DEFAULT_BACKOFF_SEC,
) -> requests.Response:
    last_exc: requests.RequestException | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=json, headers=headers, timeout=timeout)

            if response.status_code in NEVER_RETRY_STATUS:
                response.raise_for_status()
                return response

            if response.status_code == 429:
                # At most one retry for rate limits; prefer Retry-After.
                if attempt >= max_retries or attempt >= 2:
                    response.raise_for_status()
                wait = _retry_after_seconds(response)
                if wait is None:
                    wait = max(backoff_sec * 4, 8.0)
                wait = min(wait, 120.0)
                logger.warning(
                    f"HTTP 429 from {url} (attempt {attempt}/{max_retries}), "
                    f"retrying in {wait:.1f}s"
                )
                time.sleep(wait)
                continue

            if response.status_code in RETRYABLE_STATUS and attempt < max_retries:
                wait = backoff_sec * attempt
                logger.warning(
                    f"HTTP {response.status_code} from {url} "
                    f"(attempt {attempt}/{max_retries}), retrying in {wait:.1f}s"
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response
        except requests.Timeout as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            wait = backoff_sec * attempt
            logger.warning(f"Request timeout (attempt {attempt}/{max_retries}), retrying in {wait:.1f}s")
            time.sleep(wait)
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status in NEVER_RETRY_STATUS:
                raise
            if status == 429:
                raise  # already retried in-loop or exhausted
            if status in RETRYABLE_STATUS and attempt < max_retries:
                wait = backoff_sec * attempt
                logger.warning(
                    f"HTTP error {status} (attempt {attempt}/{max_retries}), retrying in {wait:.1f}s"
                )
                time.sleep(wait)
                last_exc = exc
                continue
            raise
        except requests.RequestException as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in NEVER_RETRY_STATUS:
                raise
            if status in RETRYABLE_STATUS and attempt < max_retries:
                wait = backoff_sec * attempt
                logger.warning(
                    f"HTTP error {status} (attempt {attempt}/{max_retries}), retrying in {wait:.1f}s"
                )
                time.sleep(wait)
                continue
            raise

    if last_exc:
        raise last_exc
    raise requests.RequestException("post_with_retry exhausted without response")
