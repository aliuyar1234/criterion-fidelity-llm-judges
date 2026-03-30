from __future__ import annotations

import unicodedata

_PUNCTUATION_STRIP_CHARS = ".,;:!?"
_QUOTE_STRIP_CHARS = "\"'"


def normalize_text_v1(value: str) -> str:
    """Apply the locked v1 builder-side text normalization."""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.lower()
    normalized = normalized.strip()
    normalized = " ".join(normalized.split())
    normalized = normalized.strip(_PUNCTUATION_STRIP_CHARS)
    normalized = normalized.strip(_QUOTE_STRIP_CHARS)
    return normalized


def token_count_v1(value: str) -> int:
    """Count whitespace-delimited tokens on normalized text."""

    normalized = normalize_text_v1(value)
    if not normalized:
        return 0
    return len(normalized.split(" "))


def char_length_v1(value: str) -> int:
    """Return normalized-text character length."""

    return len(normalize_text_v1(value))
