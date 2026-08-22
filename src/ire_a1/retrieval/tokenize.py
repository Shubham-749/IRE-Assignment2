"""Shared word tokenizer -- used identically for indexing article text and for
building queries, so index terms and query terms are directly comparable.
"""

import re

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _WORD_RE.findall(text.lower())
