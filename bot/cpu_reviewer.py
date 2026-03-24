"""CPU-based reviewer using quantized models via llama.cpp"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from bot.config import Config
from bot.diff_parser import ParsedDiff

LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp"
MODEL_URLS = {
    "llama-7b-q4": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
    "mistral-7b-q4": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    "codellama-7b-q4": "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf",
}


@dataclass
class ReviewIssue:
    severity: str
    file: str | None = None
    line: int | None = None
    message: str = ""
    rule: str | None = None
    suggestion: str | None = None


@dataclass
class ReviewResult:
    summary: str
    issues: list[ReviewIssue]
    recommendations: list[str]
    score: float
    model: str
    review_type: str = "cpu"

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
            "model": self.model,
            "review_type": self.review_type,
        }


class CPUReviewer:

    def __init__(self, config: Config):
        self.config = config
        self.model_path = self._get_model_path()
        self.prompt_template = self._load_prompt_template()

    def _get_model_path(self) -> Path | None:
        if self.config.model.path:
            path = Path(self.config.model.path)
            if path.exists():
                return path

        model_dir = Path.home() / ".cache" / "sorge" / "models"
        model_name = self.config.model.name

        default_path = model_dir / f"{model_name}.gguf"
        if default_path.exists():
            return default_path

        return None

    def _load_prompt_template(self) -> str:
        template_path = Path(__file__).parent / "prompts" / "review_template.txt"
        if template_path.exists():
            return template_path.read_text()

        return self._get_default_template()

    def _get_default_template(self) -> str:
        return """You are a code reviewer analyzing a pull request diff.

## Instructions
Analyze the code changes and provide feedback on:
1. Potential bugs or issues
2. Security vulnerabilities
3. Performance concerns
4. Code quality improvements
5. Best practices

## Diff to Review
{diff}

## Response Format
Provide a JSON response with:
{{
  "summary": "Brief summary of the changes",
  "issues": [
    {{
      "severity": "high|medium|low",
      "file": "filename if applicable",
      "line": line_number if applicable,
      "message": "Issue description",
      "suggestion": "Optional fix suggestion"
    }}
  ],
  "recommendations": ["List of improvement suggestions"],
  "score": 0-10 rating of code quality
}}

Be concise and focus on the most important issues."""

    def review(self, diff: ParsedDiff) -> ReviewResult:
        if not self.model_path:
            logger.warning("No model found - using heuristic review")
            return self._heuristic_review(diff)

        try:
            return self._llama_review(diff)
        except Exception as e:
            logger.error(f"Llama review failed: {e}")
            return self._heuristic_review(diff)

    def _llama_review(self, diff: ParsedDiff) -> ReviewResult:
        prompt = self.prompt_template.format(diff=diff.raw[:8000])

        cmd = [
            "llama-cli",
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", "512",
            "--temp", "0.3",
            "-t", str(self.config.model.threads),
            "--no-display-prompt",
        ]

        logger.debug(f"Running llama: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            logger.error(f"Llama failed: {result.stderr}")
            return self._heuristic_review(diff)

        return self._parse_llama_output(result.stdout, diff)

    def _parse_llama_output(self, output: str, diff: ParsedDiff) -> ReviewResult:
        import json

        try:
            data = json.loads(output)

            issues = [
                ReviewIssue(
                    severity=i.get("severity", "medium"),
                    file=i.get("file"),
                    line=i.get("line"),
                    message=i.get("message", ""),
                    suggestion=i.get("suggestion"),
                )
                for i in data.get("issues", [])
            ]

            return ReviewResult(
                summary=data.get("summary", "Review complete"),
                issues=issues,
                recommendations=data.get("recommendations", []),
                score=data.get("score", 7.0),
                model=self.config.model.name,
                review_type="cpu"
            )
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM output as JSON")
            return ReviewResult(
                summary="Review generated (format parsing failed)",
                issues=[],
                recommendations=[output[:500]],
                score=7.0,
                model=self.config.model.name,
                review_type="cpu"
            )

    def _heuristic_review(self, diff: ParsedDiff) -> ReviewResult:
        issues: list[ReviewIssue] = []
        recommendations: list[str] = []

        for filename, change in diff.file_changes.items():
            if ".test." in filename or "_test." in filename:
                continue

            if change.additions > 100:
                issues.append(ReviewIssue(
                    severity="medium",
                    file=filename,
                    message=f"Large addition ({change.additions} lines) - consider breaking into smaller changes",
                ))
                recommendations.append("Consider splitting large files into smaller, focused modules")

            if change.deletions > 50 and change.additions == 0:
                issues.append(ReviewIssue(
                    severity="low",
                    file=filename,
                    message="Large deletion without additions - ensure this is intentional",
                ))

        if diff.lines_added > diff.lines_deleted * 3:
            recommendations.append("Review ratio of additions to deletions - high ratio may indicate copy-paste patterns")

        if len(diff.files) > 10:
            issues.append(ReviewIssue(
                severity="low",
                file=None,
                message=f"Many files changed ({len(diff.files)}) - ensure changes are related and focused",
            ))

        score = 8.0
        if len(issues) > 5:
            score -= 1
        if issues and any(i.severity == "high" for i in issues):
            score -= 2

        return ReviewResult(
            summary=diff.get_summary(),
            issues=issues,
            recommendations=recommendations if recommendations else ["Code looks good - no major issues detected"],
            score=max(score, 1.0),
            model="heuristic",
            review_type="cpu"
        )


def download_model(model_name: str, target_dir: Path | None = None) -> Path:
    if model_name not in MODEL_URLS:
        raise ValueError(f"Unknown model: {model_name}")

    target_dir = target_dir or Path.home() / ".cache" / "sorge" / "models"
    target_dir.mkdir(parents=True, exist_ok=True)

    url = MODEL_URLS[model_name]
    filename = url.split("/")[-1]
    target_path = target_dir / filename

    if target_path.exists():
        logger.info(f"Model already exists at {target_path}")
        return target_path

    logger.info(f"Downloading {model_name} from {url}")

    import requests

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = (downloaded / total_size) * 100
                    print(f"\rDownloading: {pct:.1f}%", end="", flush=True)

    print()
    logger.info(f"Model saved to {target_path}")

    return target_path
