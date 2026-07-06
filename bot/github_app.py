"""GitHub App installation token helpers."""

from __future__ import annotations

import os
import time

import requests
from ghapi.all import GhApi
from loguru import logger


def _load_private_key(pem: str | None) -> str:
    if pem:
        # Handle both actual \n chars and literal "\\n" strings
        pem = pem.replace("\\n", "\n")
        # Strip trailing whitespace from each line
        lines = [line.strip() for line in pem.split("\n")]
        # Remove any empty lines at start/end but keep structure
        key = "\n".join(lines)
        return key
    path = os.getenv("SORGE_APP_PRIVATE_KEY_PATH")
    if path and os.path.isfile(path):
        return open(path, encoding="utf-8").read().strip()
    return ""


def create_app_jwt(app_id: str, private_key_pem: str) -> str:
    """Build a short-lived JWT for GitHub App authentication."""
    try:
        import jwt
    except ImportError as e:
        raise RuntimeError("PyJWT is required for GitHub App auth: pip install PyJWT[crypto]") from e

    now = int(time.time())
    payload = {
        "iat": now - 60,
        "exp": now + 600,
        "iss": app_id,
    }

    key = private_key_pem.strip()
    # Debug: log PEM shape (safe - first/last lines are known markers)
    first_line = key.split("\n")[0] if key else "EMPTY"
    last_line = key.split("\n")[-1] if key else "EMPTY"
    logger.debug(
        f"PEM first: {first_line} | last: {last_line} | "
        f"chars: {len(key)} | lines: {len(key.split(chr(10)))}"
    )

    return jwt.encode(payload, key, algorithm="RS256")


def get_installation_token(
    installation_id: int | str,
    *,
    app_id: str | None = None,
    private_key: str | None = None,
) -> str | None:
    """Exchange App JWT for an installation access token."""
    app_id = app_id or os.getenv("SORGE_APP_ID", "")
    pem = _load_private_key(private_key or os.getenv("SORGE_APP_PRIVATE_KEY"))

    if not app_id or not pem:
        logger.debug("GitHub App credentials not configured")
        return None

    try:
        app_jwt = create_app_jwt(app_id, pem)
    except Exception as e:
        logger.error(f"Failed to create App JWT: {e}")
        return None

    try:
        api = GhApi(token=app_jwt)
        resp = api.apps.create_installation_access_token(installation_id)
        return resp.token
    except Exception as e:
        logger.error(f"Failed to get installation token: {e}")
        return None


def fetch_pr_diff(repo: str, pr_number: int, token: str) -> str:
    """Fetch unified diff for a PR.

    Uses raw requests because ghapi's REST client expects JSON responses,
    while this endpoint needs Accept: application/vnd.github.v3.diff
    which returns raw text instead of JSON.
    """
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.diff",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.text