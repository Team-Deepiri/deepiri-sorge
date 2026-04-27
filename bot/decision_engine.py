"""Decision engine for determining review strategy"""

import re
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from bot.config import Config
from bot.diff_parser import ParsedDiff


class Action(Enum):
    SKIP = "skip"
    CPU_REVIEW = "cpu_review"
    GPU_REVIEW = "gpu_review"
    GITHUB_MODELS = "github_models"
    GEMINI = "gemini"


@dataclass
class ReviewDecision:
    action: Action
    reason: str
    confidence: float = 1.0
    skip_category: str | None = None


class DecisionEngine:

    DOCS_PATTERNS = [
        r"\.md$",
        r"\.rst$",
        r"\.txt$",
        r"docs?/",
        r"documentation/",
        r"CHANGELOG",
        r"LICENSE",
        r"README",
    ]

    DEPS_PATTERNS = [
        r"package-lock\.json$",
        r"yarn\.lock$",
        r"poetry\.lock$",
        r"pipfile\.lock$",
        r"requirements\.txt$",
        r"Gemfile\.lock$",
        r"Cargo\.lock$",
        r"go\.mod$",
        r"go\.sum$",
        r"\.gemspec$",
        r"pubspec\.yaml$",
        r"Podfile\.lock$",
        r"composer\.lock$",
    ]

    TEST_PATTERNS = [
        r"_test\.py$",
        r"_tests\.py$",
        r"test_.*\.py$",
        r".*_test\.go$",
        r".*_test\.js$",
        r".*_test\.ts$",
        r".*\.spec\.(js|ts)$",
        r".*\.test\.(js|ts)$",
        r"__tests__/",
        r"/tests/",
    ]

    def __init__(self, config: Config):
        self.config = config

    def decide(self, diff: ParsedDiff) -> ReviewDecision:
        if not self.config.sorge.get("enabled", True):
            return ReviewDecision(
                action=Action.SKIP,
                reason="Sorge is disabled in config",
                skip_category="disabled"
            )

        if self._is_docs_only(diff):
            return ReviewDecision(
                action=Action.SKIP,
                reason="Docs-only changes - no review needed",
                skip_category="docs"
            )

        if self.config.filters.skip_deps and self._is_deps_only(diff):
            return ReviewDecision(
                action=Action.SKIP,
                reason="Dependency changes only - no review needed",
                skip_category="deps"
            )

        if self.config.filters.skip_tests and self._is_tests_only(diff):
            return ReviewDecision(
                action=Action.SKIP,
                reason="Test-only changes - no review needed",
                skip_category="tests"
            )

        total_lines = diff.lines_added + diff.lines_deleted

        if total_lines < self.config.filters.min_lines:
            return ReviewDecision(
                action=Action.SKIP,
                reason=f"Too few lines changed ({total_lines} < {self.config.filters.min_lines})",
                skip_category="too_small"
            )

        estimated_tokens = self._estimate_tokens(diff)

        if self.config.github_models.enabled and self.config.gemini.enabled:
            if estimated_tokens <= self.config.routing.small_pr_threshold:
                return ReviewDecision(
                    action=Action.GITHUB_MODELS,
                    reason=f"Small diff (~{estimated_tokens} tokens) - using GitHub Models",
                    confidence=0.95
                )
            elif estimated_tokens > self.config.routing.large_pr_threshold:
                return ReviewDecision(
                    action=Action.GEMINI,
                    reason=f"Large diff (~{estimated_tokens} tokens) - using Gemini 2.5 Pro",
                    confidence=0.95
                )
            else:
                return ReviewDecision(
                    action=Action.GITHUB_MODELS,
                    reason=f"Medium diff (~{estimated_tokens} tokens) - using GitHub Models",
                    confidence=0.9
                )

        if self.config.github_models.enabled:
            return ReviewDecision(
                action=Action.GITHUB_MODELS,
                reason=f"Using GitHub Models ({total_lines} lines)",
                confidence=0.95
            )

        if self.config.gemini.enabled:
            return ReviewDecision(
                action=Action.GEMINI,
                reason=f"Using Gemini 2.5 Pro ({total_lines} lines)",
                confidence=0.95
            )

        if self.config.gpu.enabled and total_lines > self.config.gpu.threshold_lines:
            return ReviewDecision(
                action=Action.GPU_REVIEW,
                reason=f"Large diff ({total_lines} lines) - using GPU",
                confidence=0.9
            )

        if total_lines > self.config.filters.max_cpu_lines:
            if self.config.gpu.enabled:
                return ReviewDecision(
                    action=Action.GPU_REVIEW,
                    reason=f"Exceeds CPU limit ({total_lines} > {self.config.filters.max_cpu_lines}) - GPU",
                    confidence=0.8
                )
            else:
                logger.warning("Diff exceeds CPU limit but GPU disabled - running limited CPU review")

        return ReviewDecision(
            action=Action.CPU_REVIEW,
            reason=f"Standard review ({total_lines} lines)",
            confidence=0.95
        )

    def _estimate_tokens(self, diff: ParsedDiff) -> int:
        raw_length = len(diff.raw)
        return raw_length // 4

    def _matches_any_pattern(self, filename: str, patterns: list[str], flags: int = 0) -> bool:
        return any(re.search(p, filename, flags) for p in patterns)

    def _is_docs_only(self, diff: ParsedDiff) -> bool:
        if not self.config.filters.skip_docs or not diff.files:
            return False
        return all(self._matches_any_pattern(f, self.DOCS_PATTERNS, re.IGNORECASE) for f in diff.files)

    def _is_deps_only(self, diff: ParsedDiff) -> bool:
        if not diff.files:
            return False
        return all(self._matches_any_pattern(f, self.DEPS_PATTERNS) for f in diff.files)

    def _is_tests_only(self, diff: ParsedDiff) -> bool:
        if not diff.files:
            return False
        return all(self._matches_any_pattern(f, self.TEST_PATTERNS) for f in diff.files)

    def get_complexity_score(self, diff: ParsedDiff) -> float:
        score = 0.0

        score += min(diff.lines_added / 100, 5)
        score += min(diff.lines_deleted / 100, 5)
        score += min(len(diff.files) / 10, 3)

        for file in diff.files:
            if any(x in file.lower() for x in ["api", "core", "service", "handler"]):
                score += 0.5

        return min(score, 10.0)
