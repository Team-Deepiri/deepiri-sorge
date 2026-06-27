"""Tests for formatting helpers and review comment rendering."""

from __future__ import annotations

from tests.helpers import install_loguru_stub

install_loguru_stub()

from bot.comment_poster import CommentPoster
from bot.runners.base import ReviewIssue, ReviewResult
from bot.utils.formatting import (
    chunk_text,
    clean_multiline_text,
    format_blockquote,
    format_issue_location,
    normalize_whitespace,
)


class TestFormattingHelpers:
    def test_normalize_whitespace_collapses_spacing(self):
        assert normalize_whitespace("  a   b\tc  ") == "a b c"

    def test_clean_multiline_text_preserves_lines(self):
        text = "  first line  \nsecond line   \n\n"
        assert clean_multiline_text(text) == "first line\nsecond line"

    def test_format_issue_location_with_file_and_line(self):
        assert format_issue_location("  src/app.py  ", 17) == "**src/app.py**:17"

    def test_format_issue_location_without_file(self):
        assert format_issue_location(None, None) == "General"

    def test_format_blockquote_handles_multiline_text(self):
        text = " first line \n\n second line "
        assert format_blockquote(text) == "> first line\n>\n>  second line"

    def test_chunk_text_splits_on_word_boundaries(self):
        chunks = chunk_text("alpha beta gamma delta", max_length=10)
        assert chunks == ["alpha beta", "gamma", "delta"]


class TestCommentPosterFormatting:
    def test_format_review_comment_renders_sections_and_normalizes_text(self):
        review = ReviewResult(
            summary=" Summary line  \nextra details   ",
            issues=[
                ReviewIssue(
                    severity="medium",
                    file="  src/service.py ",
                    line=12,
                    message=" First line \nsecond line ",
                    suggestion=" use a helper  method ",
                )
            ],
            recommendations=[" tighten spacing  ", " add tests "],
            score=8.5,
            model=" heuristic ",
            review_type=" groq ",
        )

        body = CommentPoster()._format_review_comment(review)

        assert "**Model:** heuristic (groq)" in body
        assert "### Summary\nSummary line\nextra details" in body
        assert ":warning: **src/service.py**:12" in body
        assert "> First line\n> second line" in body
        assert "> _Suggestion: use a helper method_" in body
        assert "- tighten spacing" in body
        assert "- add tests" in body

    def test_format_review_comment_uses_general_for_missing_location(self):
        review = ReviewResult(
            summary="Looks fine",
            issues=[ReviewIssue(severity="low", file=None, line=None, message="Watch this area")],
            recommendations=[],
            score=9.0,
            model="heuristic",
        )

        body = CommentPoster()._format_review_comment(review)

        assert ":information_source: General" in body
        assert "> Watch this area" in body
