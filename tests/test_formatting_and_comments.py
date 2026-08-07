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
            latency_ms=0.0,

            review_type=" groq ",
        )

        body = CommentPoster()._format_review_comment(review)

        assert "<!-- sorge-review-anchor -->" in body
        assert "**Model:** heuristic (groq)" in body
        assert "### Summary\nSummary line\nextra details" in body
        assert ":warning: **src/service.py**:12" in body
        assert "> First line\n> second line" in body
        assert "> _Suggestion: use a helper method_" in body
        assert "- tighten spacing" in body
        assert "- add tests" in body

    def test_format_review_comment_shows_parse_warning(self):
        review = ReviewResult(
            summary="Partial output",
            issues=[],
            recommendations=[],
            score=5.0,
            model="gemma",
            latency_ms=0.0,
            review_type="openrouter",
            parse_warning="non_json_response",
        )

        body = CommentPoster()._format_review_comment(review)

        assert "Review incomplete" in body
        assert "non_json_response" in body

    def test_format_review_comment_uses_general_for_missing_location(self):
        review = ReviewResult(
            summary="Looks fine",
            issues=[ReviewIssue(severity="low", file=None, line=None, message="Watch this area")],
            recommendations=[],
            latency_ms=0.0,

            score=9.0,
            model="heuristic",
        )

        body = CommentPoster()._format_review_comment(review)

        assert ":information_source: General" in body
        assert "> Watch this area" in body

    def test_format_rate_limited_comment_omits_quality_score(self):
        review = ReviewResult(
            summary="Review deferred — free-tier provider rate limits were hit.",
            issues=[],
            recommendations=["Wait a few minutes, then comment `/sorge` again"],
            score=0.0,
            model="none",
            latency_ms=0.0,
            review_type="rate_limited",
        )

        body = CommentPoster()._format_review_comment(review)

        assert "Temporarily unavailable" in body or "Deferred" in body
        assert "Quality Score" not in body
        assert "needs redesign" not in body
        assert "/sorge" in body
        assert "not a code review result" in body

    def test_format_soft_quota_comment_is_distinct_from_http_429(self):
        soft = ReviewResult(
            summary="Gemini soft daily quota is exhausted.",
            issues=[],
            recommendations=["Wait for UTC day roll"],
            score=0.0,
            model="none",
            latency_ms=0.0,
            review_type="soft_quota_exhausted",
            routing_meta={
                "deferral_class": "soft_quota_exhausted",
                "quota": {"gemini": {"limit": 20, "used": 20, "remaining": 0}},
            },
        )
        http = ReviewResult(
            summary="Provider HTTP 429 rate limits blocked the review.",
            issues=[],
            recommendations=["Wait for 429 cooldowns"],
            score=0.0,
            model="none",
            latency_ms=0.0,
            review_type="http_429",
            routing_meta={"deferral_class": "http_429"},
        )
        soft_body = CommentPoster()._format_review_comment(soft)
        http_body = CommentPoster()._format_review_comment(http)
        assert "soft daily quota" in soft_body.lower()
        assert "HTTP 429" in http_body
        assert soft_body != http_body
        assert "`soft_quota_exhausted`" in soft_body
        assert "`http_429`" in http_body

    def test_format_vacuous_comment_is_distinct(self):
        review = ReviewResult(
            summary="Providers returned truncated or empty review JSON.",
            issues=[],
            recommendations=["Re-run /sorge"],
            score=0.0,
            model="none",
            latency_ms=0.0,
            review_type="vacuous_or_truncated",
            routing_meta={"deferral_class": "vacuous_or_truncated"},
        )
        body = CommentPoster()._format_review_comment(review)
        assert "truncated/empty" in body.lower() or "vacuous" in body.lower()
        assert "not RPM" in body or "vacuous_or_truncated" in body
        assert "Quality Score" not in body

    def test_post_review_edits_preferred_or_posts_new(self, monkeypatch):
        poster = CommentPoster("token")
        calls: list[tuple] = []

        def fake_upsert(repo, pr, body, *, preferred_comment_id=None, reuse_previous=False):
            calls.append(("upsert", repo, pr, preferred_comment_id, reuse_previous))
            return 99

        def fake_post(repo, pr, body, commit_id=None):
            calls.append(("post", repo, pr))
            return 100

        monkeypatch.setattr(poster, "upsert_comment", fake_upsert)
        monkeypatch.setattr(poster, "post_comment", fake_post)

        review = ReviewResult(
            summary="ok",
            issues=[],
            recommendations=[],
            score=8.0,
            model="m",
            latency_ms=1.0,
            review_type="groq",
        )
        # No preferred id → new post (per-run hybrid; do not rewrite prior runs).
        assert poster.post_review("org/r", 7, review) == 100
        assert calls == [("post", "org/r", 7)]

        calls.clear()
        assert poster.post_review("org/r", 7, review, preferred_comment_id=55) == 99
        assert calls == [("upsert", "org/r", 7, 55, False)]

        calls.clear()
        assert poster.post_review("org/r", 7, review, edit_existing=False) == 100
        assert calls == [("post", "org/r", 7)]

    def test_upsert_comment_skips_previous_by_default(self, monkeypatch):
        poster = CommentPoster("token")
        monkeypatch.setattr(poster, "get_previous_review", lambda *a, **k: 11)
        edited: list[int] = []

        def fake_update(repo, cid, body):
            edited.append(cid)
            return True

        monkeypatch.setattr(poster, "update_comment", fake_update)
        monkeypatch.setattr(poster, "post_comment", lambda *a, **k: 100)

        assert poster.upsert_comment("org/r", 1, "## Sorge AI Code Review\n\nx") == 100
        assert edited == []

        assert (
            poster.upsert_comment(
                "org/r",
                1,
                "## Sorge AI Code Review\n\nx",
                preferred_comment_id=55,
            )
            == 55
        )
        assert edited == [55]

    def test_append_escalation_preserves_existing_review(self, monkeypatch):
        """Draining a deferred ticket must not wipe out the rest of an
        already-posted multi-chunk review — it should append, not replace."""
        from dataclasses import dataclass

        @dataclass
        class FakeTicket:
            ticket_id: str
            reason: str
            groq_score: float
            files: list

        poster = CommentPoster("token")
        original_body = (
            "<!-- sorge-review-anchor -->\n"
            "## Sorge AI Code Review\n\n"
            "### Issues Found\n\n"
            ":x: `other_file.py:10`\n> some other finding\n\n"
            "ℹ️ `deferred_file.py`\n\n"
            "> Chunk skipped by scheduler: no eligible provider\n\n"
            "---\n\n"
            "*Review generated by [deepiri-sorge](https://github.com/deepiri/deepiri-sorge)*"
        )
        monkeypatch.setattr(poster, "get_comment_body", lambda repo, cid: original_body)

        patched: list[tuple] = []

        def fake_update(repo, cid, body):
            patched.append((repo, cid, body))
            return True

        monkeypatch.setattr(poster, "update_comment", fake_update)

        ticket = FakeTicket(
            ticket_id="abc123",
            reason="security",
            groq_score=9.5,
            files=["deferred_file.py"],
        )
        result = ReviewResult(
            summary="Deep dive found a real issue.",
            issues=[
                ReviewIssue(
                    severity="high",
                    file="deferred_file.py",
                    line=42,
                    message="Actual security problem.",
                    rule="",
                    suggestion="Fix it.",
                )
            ],
            recommendations=["Do the thing."],
            score=4.0,
            model="gemini",
            latency_ms=1.0,
            review_type="gemini",
        )

        ok = poster.append_escalation("org/r", 123, ticket, result)
        assert ok is True
        assert len(patched) == 1
        _, cid, new_body = patched[0]
        assert cid == 123
        # Original findings and skip-record must survive.
        assert "some other finding" in new_body
        assert "Chunk skipped by scheduler" in new_body
        # New escalated finding is appended, not swapped in.
        assert "Actual security problem." in new_body
        assert "Deep dive found a real issue." in new_body
        # Footer stays at the very end rather than being duplicated/lost.
        assert new_body.rstrip().endswith(
            "*Review generated by [deepiri-sorge](https://github.com/deepiri/deepiri-sorge)*"
        )

    def test_append_escalation_is_idempotent(self, monkeypatch):
        """Re-draining the same ticket (e.g. after a retry) shouldn't append twice."""
        from dataclasses import dataclass

        @dataclass
        class FakeTicket:
            ticket_id: str
            reason: str
            groq_score: float
            files: list

        poster = CommentPoster("token")
        already_applied = (
            "<!-- sorge-review-anchor -->\n## Sorge AI Code Review\n\n"
            "<!-- sorge-escalation-abc123 -->\n### Escalated finding\n\n"
            "*Review generated by [deepiri-sorge](https://github.com/deepiri/deepiri-sorge)*"
        )
        monkeypatch.setattr(poster, "get_comment_body", lambda repo, cid: already_applied)
        calls: list = []
        monkeypatch.setattr(poster, "update_comment", lambda *a, **k: calls.append(1) or True)

        ticket = FakeTicket(ticket_id="abc123", reason="security", groq_score=9.5, files=["x.py"])
        result = ReviewResult(
            summary="s", issues=[], recommendations=[], score=5.0,
            model="gemini", latency_ms=1.0, review_type="gemini",
        )
        assert poster.append_escalation("org/r", 123, ticket, result) is True
        assert calls == []  # no PATCH issued — already applied
