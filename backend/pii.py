"""PII masking guardrail (Phase 9, §9.3).

Payment data is already masked at the source (only last4 is stored/returned).
This is defense-in-depth for anything that gets logged: if a card-like number
ever appears in a string we log, collapse it to "card ending 1234".
"""

from __future__ import annotations

import re

# 13-16 digits with optional single space/dash separators BETWEEN digits only
# (no trailing separator consumed). Covers most card formats.
_CARD_RE = re.compile(r"\b\d(?:[ -]?\d){12,15}\b")


def mask_pii(text: str) -> str:
    if not text:
        return text

    def _repl(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 16:
            return f"card ending {digits[-4:]}"
        return m.group()

    return _CARD_RE.sub(_repl, str(text))
