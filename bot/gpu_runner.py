"""GPU runner for complex PRs via serverless endpoints"""

import os
import time
from dataclasses import dataclass

import requests
from loguru import logger

from bot.config import Config
from bot.cpu_reviewer import ReviewIssue
from bot.diff_parser import ParsedDiff


@dataclass
class GPUReviewResult:
    summary: str
    issues: list[ReviewIssue]
    recommendations: list[str]
    score: float
    endpoint: str
    latency_ms: float
    review_type: str = "gpu"

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "issues": [
                {
                    "severity": i.severity,
                    "file": i.file,
                    "line": i.line,
                    "message": i.message,
                    "rule": i.rule,
                    "suggestion": i.suggestion,
                }
                for i in self.issues
            ],
            "recommendations": self.recommendations,
            "score": self.score,
            "endpoint": self.endpoint,
            "latency_ms": self.latency_ms,
            "review_type": self.review_type,
        }


class GPURunner:

    def __init__(self, config: Config):
        self.config = config
        self.endpoint = self.config.gpu.endpoint
        self.api_key = self.config.gpu.api_key or os.getenv("GPU_API_KEY", "")
        self.timeout = self.config.gpu.timeout

    def is_available(self) -> bool:
        return bool(self.endpoint)

    def review(self, diff: ParsedDiff) -> GPUReviewResult | None:
        if not self.is_available():
            logger.warning("GPU endpoint not configured")
            return None

        import time
        start_time = time.time()

        try:
            return self._call_endpoint(diff, start_time)
        except requests.Timeout:
            logger.error("GPU request timed out")
            return self._timeout_result(diff, start_time)
        except requests.RequestException as e:
            logger.error(f"GPU request failed: {e}")
            return None

    def _call_endpoint(self, diff: ParsedDiff, start_time: float) -> GPUReviewResult:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "diff": diff.raw,
            "files": diff.files,
            "language_counts": diff.language_counts,
            "lines_added": diff.lines_added,
            "lines_deleted": diff.lines_deleted,
            "files_changed": diff.files_changed,
            "options": {
                "style": self.config.review.style,
                "include_security": self.config.review.include_security,
                "include_performance": self.config.review.include_performance,
                "include_style": self.config.review.include_style,
            }
        }

        logger.debug(f"Calling GPU endpoint: {self.endpoint}")

        response = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

        response.raise_for_status()

        latency_ms = (time.time() - start_time) * 1000
        data = response.json()

        issues = [
            ReviewIssue(
                severity=i.get("severity", "medium"),
                file=i.get("file"),
                line=i.get("line"),
                message=i.get("message", ""),
                rule=i.get("rule"),
                suggestion=i.get("suggestion"),
            )
            for i in data.get("issues", [])
        ]

        return GPUReviewResult(
            summary=data.get("summary", "GPU review complete"),
            issues=issues,
            recommendations=data.get("recommendations", []),
            score=data.get("score", 7.0),
            endpoint=self.endpoint,
            latency_ms=latency_ms,
        )

    def _timeout_result(self, diff: ParsedDiff, start_time: float) -> GPUReviewResult:
        latency_ms = (time.time() - start_time) * 1000

        return GPUReviewResult(
            summary="GPU review timed out - consider splitting into smaller PRs",
            issues=[
                ReviewIssue(
                    severity="low",
                    file=None,
                    message="Review timed out - the diff may be too large for timely analysis",
                    suggestion="Consider breaking this PR into smaller, focused changes",
                )
            ],
            recommendations=[
                "Split large PRs into smaller, focused changes",
                "Use CPU review for faster turnaround on simple changes",
            ],
            score=5.0,
            endpoint=self.endpoint,
            latency_ms=latency_ms,
        )


class RunPodRunner(GPURunner):

    def __init__(self, config: Config, endpoint_id: str):
        super().__init__(config)
        self.endpoint_id = endpoint_id
        self.endpoint = f"https://api.runpod.io/v2/{endpoint_id}/run"

    def _call_endpoint(self, diff: ParsedDiff, start_time: float) -> GPUReviewResult:
        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "input": {
                "diff": diff.raw[:32000],
                "options": {
                    "style": self.config.review.style,
                }
            }
        }

        response = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=self.timeout + 10,
        )

        response.raise_for_status()
        run_result = response.json()

        poll_url = run_result.get("status", {}).get("poll_url")
        if not poll_url:
            raise ValueError("No poll URL in RunPod response")

        result = self._poll_runpod(poll_url, headers)
        latency_ms = (time.time() - start_time) * 1000

        return GPUReviewResult(
            summary=result.get("summary", "Review complete"),
            issues=[],
            recommendations=result.get("recommendations", []),
            score=result.get("score", 7.0),
            endpoint=self.endpoint,
            latency_ms=latency_ms,
        )

    def _poll_runpod(self, poll_url: str, headers: dict) -> dict:
        import time

        max_polls = 60
        poll_interval = 2

        for _ in range(max_polls):
            response = requests.get(poll_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            status = data.get("status", {})

            if status == "COMPLETED":
                return data.get("output", {})
            elif status in ("FAILED", "CANCELLED"):
                raise ValueError(f"RunPod task {status}")

            time.sleep(poll_interval)

        raise TimeoutError("RunPod polling timed out")


class VastAIRunner(GPURunner):

    def __init__(self, config: Config, api_key: str):
        super().__init__(config)
        self.vast_api_key = api_key
        self.base_url = "https://console.vast.ai/api/v0"

    def _call_endpoint(self, diff: ParsedDiff, start_time: float) -> GPUReviewResult:
        headers = {
            "Authorization": f"Bearer {self.vast_api_key}",
        }

        payload = {
            "prompt": f"Analyze this code diff:\n\n{diff.raw[:32000]}",
            "max_tokens": 1024,
            "temperature": 0.3,
        }

        response = requests.post(
            f"{self.base_url}/inference",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

        response.raise_for_status()

        latency_ms = (time.time() - start_time) * 1000
        data = response.json()

        return GPUReviewResult(
            summary="Deep analysis complete",
            issues=[],
            recommendations=[data.get("text", "Analysis complete")[:500]],
            score=7.0,
            endpoint=self.endpoint,
            latency_ms=latency_ms,
        )
