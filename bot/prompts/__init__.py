"""Prompt templates for review runners."""

from pathlib import Path

_DIR = Path(__file__).parent
_TEMPLATE_PATH = _DIR / "review_template.txt"
_GROQ_BUG_DETECTOR_PATH = _DIR / "groq_bug_detector.txt"


def load_review_template() -> str:
    if _TEMPLATE_PATH.exists():
        return _TEMPLATE_PATH.read_text(encoding="utf-8")
    return ""


def load_groq_bug_detector_template() -> str:
    if _GROQ_BUG_DETECTOR_PATH.exists():
        return _GROQ_BUG_DETECTOR_PATH.read_text(encoding="utf-8")
    return load_review_template()
