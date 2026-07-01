"""GitHub App installation token helpers."""

from __future__ import annotations

import os
import time

import requests
from loguru import logger


def _load_private_key(pem: str | None) -> str:
    if pem:
        return pem.replace("\\n", "\n")
    path = os.getenv("SORGE_APP_PRIVATE_KEY_PATH")
    if path and os.path.isfile(path):
        return open(path, encoding="utf-8").read()
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
    return jwt.encode(payload, private_key_pem, algorithm="RS256")


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

    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    headers = {
        "Authorization": f"Bearer {app_jwt}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        resp = requests.post(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json().get("token")
    except requests.RequestException as e:
        logger.error(f"Failed to get installation token: {e}")
        return None


def fetch_pr_diff(repo: str, pr_number: int, token: str) -> str:
    """Fetch unified diff for a PR."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.diff",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.text
