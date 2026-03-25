"""GitHub API utilities"""

import os
from typing import Any

import requests


class GitHubAPI:

    def __init__(self, token: str | None = None):
        self.token = token or os.getenv("GITHUB_TOKEN", "")
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"

        response = requests.request(
            method,
            url,
            headers=self.headers,
            **kwargs,
        )

        response.raise_for_status()

        if response.content:
            return response.json()
        return {}

    def get_pr(self, owner: str, repo: str, pr_number: int) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/repos/{owner}/{repo}/pulls/{pr_number}",
        )

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/files",
        )

    def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}"

        headers = {**self.headers, "Accept": "application/vnd.github.v3.diff"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        return response.text

    def get_commits(self, owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/commits",
        )

    def post_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
            json={"body": body},
        )

    def get_comments(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
        )

    def delete_comment(
        self,
        owner: str,
        repo: str,
        comment_id: int,
    ) -> None:
        self._request(
            "DELETE",
            f"/repos/{owner}/{repo}/issues/comments/{comment_id}",
        )

    def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str | None = None,
    ) -> dict[str, Any]:
        params = {"ref": ref} if ref else {}
        return self._request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{path}",
            params=params,
        )

    def get_branch(
        self,
        owner: str,
        repo: str,
        branch: str,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/repos/{owner}/{repo}/branches/{branch}",
        )

    def compare_commits(
        self,
        owner: str,
        repo: str,
        base: str,
        head: str,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/repos/{owner}/{repo}/compare/{base}...{head}",
        )

    def create_check_run(
        self,
        owner: str,
        repo: str,
        name: str,
        head_sha: str,
        status: str = "in_progress",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/repos/{owner}/{repo}/check-runs",
            json={
                "name": name,
                "head_sha": head_sha,
                "status": status,
            },
        )

    def update_check_run(
        self,
        owner: str,
        repo: str,
        check_run_id: int,
        **kwargs,
    ) -> dict[str, Any]:
        return self._request(
            "PATCH",
            f"/repos/{owner}/{repo}/check-runs/{check_run_id}",
            json=kwargs,
        )
