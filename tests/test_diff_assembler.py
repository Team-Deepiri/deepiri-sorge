"""Tests for DiffAssembler (406 too_large fallback)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bot.diff_assembler import assemble_pr_diff, stitch_file_patches


def test_stitch_file_patches_builds_headers():
    files = [
        {
            "filename": "a.py",
            "status": "modified",
            "patch": "@@ -1 +1 @@\n-old\n+new",
        },
        {
            "filename": "bin.dat",
            "status": "added",
            "changes": 0,
        },
    ]
    out = stitch_file_patches(files)
    assert "diff --git a/a.py b/a.py" in out
    assert "+new" in out
    assert "no patch available" in out


def test_assemble_falls_back_on_406():
    mono = MagicMock()
    mono.status_code = 406
    mono.text = '{"message":"Sorry, the diff exceeded the maximum number of lines (20000)","errors":[{"code":"too_large"}]}'

    files_resp = MagicMock()
    files_resp.status_code = 200
    files_resp.json.return_value = [
        {"filename": "x.py", "status": "modified", "patch": "@@ -1 +1 @@\n+hi"},
    ]
    files_resp.raise_for_status = MagicMock()

    with patch("bot.diff_assembler.requests.get", side_effect=[mono, files_resp]):
        text = assemble_pr_diff("org/repo", 1, "token")
    assert "x.py" in text
    assert "+hi" in text


def test_assemble_raises_on_other_errors():
    mono = MagicMock()
    mono.status_code = 403
    mono.text = "forbidden"
    mono.raise_for_status.side_effect = Exception("403")

    with patch("bot.diff_assembler.requests.get", return_value=mono):
        with pytest.raises(Exception):
            assemble_pr_diff("org/repo", 1, "token")
