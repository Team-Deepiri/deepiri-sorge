"""Prompt templates for review runners."""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "review_template.txt"


def load_review_template() -> str:
    if _TEMPLATE_PATH.exists():
        return _TEMPLATE_PATH.read_text(encoding="utf-8")
    return ""
