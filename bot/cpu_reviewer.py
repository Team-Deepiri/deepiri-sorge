"""CPU-based reviewer using quantized models via llama.cpp"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError

from loguru import logger

from bot.config import Config
from bot.diff_parser import ParsedDiff

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None
    logger.warning("llama-cpp-python is not installed. Local LLM inference will be unavailable.")

LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp"
MODEL_URLS = {
    "llama-7b-q4": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
    "mistral-7b-q4": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    "codellama-7b-q4": "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf",
}

# --- Validation Schemas ---

class IssueSchema(BaseModel):
    severity: str = Field(default="medium", pattern="^(high|medium|low)$")
    file: str | None = None
    line: int | None = None
    message: str = Field(default="Unknown issue")
    rule: str | None = None
    suggestion: str | None = None

class ReviewOutputSchema(BaseModel):
    summary: str = Field(default="Review complete")
    issues: list[IssueSchema] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    score: float = Field(default=7.0, ge=0.0, le=10.0)


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
        self._llm_instance = None  # Cache model in memory

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
Provide ONLY a valid JSON response with:
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
"""

    def review(self, diff: ParsedDiff) -> ReviewResult:
        if not self.model_path or Llama is None:
            logger.warning("No model found or llama-cpp-python missing - using heuristic review")
            return self._heuristic_review(diff)

        try:
            return self._llama_review(diff)
        except Exception as e:
            logger.error(f"Llama review failed: {e}", exc_info=True)
            return self._heuristic_review(diff)

    def _llama_review(self, diff: ParsedDiff) -> ReviewResult:
        # Leave buffer for system prompt and output generation
        max_diff_len = max(self.config.model.context_size * 2, 8000) 
        prompt = self.prompt_template.format(diff=diff.raw[:max_diff_len])
        
        logger.debug(f"LLM Prompt generated (length: {len(prompt)})")
        logger.trace(f"Full prompt text:\n{prompt}") # Deep debug hook

        # Lazy load model
        if self._llm_instance is None:
            logger.info(f"Loading {self.config.model.name} natively into memory via llama-cpp-python...")
            self._llm_instance = Llama(
                model_path=str(self.model_path),
                n_ctx=self.config.model.context_size,
                n_threads=self.config.model.threads,
                verbose=False # Keep standard output clean
            )

        logger.info("Executing LLM inference...")
        response = self._llm_instance(
            prompt,
            max_tokens=1024,
            temperature=0.2, # Low temperature for more deterministic JSON
            stop=["```\n", "}\n\n"]
        )

        output_text = response['choices'][0]['text'].strip()
        logger.debug(f"Raw LLM output received (length: {len(output_text)})")
        logger.trace(f"Raw LLM Text:\n{output_text}") # Deep debug hook
        
        return self._parse_llama_output(output_text, diff)

    def _parse_llama_output(self, output: str, diff: ParsedDiff) -> ReviewResult:
        # Attempt to strip markdown formatting if the model wrapped it in ```json
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            output = json_match.group(0)

        try:
            data = json.loads(output)
            # Schema Validation via Pydantic
            validated = ReviewOutputSchema(**data)

            issues = [
                ReviewIssue(
                    severity=i.severity,
                    file=i.file,
                    line=i.line,
                    message=i.message,
                    suggestion=i.suggestion,
                )
                for i in validated.issues
            ]

            logger.info(f"Successfully validated review with {len(issues)} issues found.")
            return ReviewResult(
                summary=validated.summary,
                issues=issues,
                recommendations=validated.recommendations,
                score=validated.score,
                model=self.config.model.name,
                review_type="cpu"
            )

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse or validate LLM JSON output: {e}")
            logger.debug(f"Problematic output segment: {output[:500]}")
            
            return ReviewResult(
                summary="Review generation succeeded, but result formatting was corrupted.",
                issues=[],
                recommendations=["Consider adjusting the LLM prompt for stricter JSON output."],
                score=7.0,
                model=self.config.model.name,
                review_type="cpu-error"
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