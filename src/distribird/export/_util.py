"""Shared helpers for code/text exporters."""

from __future__ import annotations

import re


def safe_identifier(name: str) -> str:
    """Turn an arbitrary parameter name into a valid Python/R identifier.

    Replaces every character outside ``[0-9A-Za-z_]`` with ``_`` and prefixes a
    leading underscore when the result is empty or starts with a digit. The
    output is a legal variable name in both Python and R, so generated scripts
    never raise ``SyntaxError`` on names like ``"Vcmax (25C)"`` or ``"3PG_rate"``.
    """
    s = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not s or s[0].isdigit():
        s = "_" + s
    return s


def comment_safe(text: str) -> str:
    """Collapse a string to a single line for safe embedding in a ``#`` comment.

    Newlines in externally-sourced text (paper titles, prior rationale) would
    otherwise terminate the comment and turn the remainder into uncommented code.
    """
    return re.sub(r"\s+", " ", text).strip()
