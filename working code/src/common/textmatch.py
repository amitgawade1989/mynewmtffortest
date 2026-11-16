"""Deterministic text matching used to verify that quotes really exist in a chunk."""

import re
import unicodedata
from difflib import SequenceMatcher


def normalize(text):
    """Lowercase, unify quotes/dashes, drop markdown table pipes and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("—", "-").replace("–", "-").replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()


def quote_score(quote, text):
    """How well `quote` appears inside `text`: 1.0 = exact (after normalising), else best fuzzy window ratio."""
    q, t = normalize(quote), normalize(text)
    if not q:
        return 0.0
    if q in t:
        return 1.0
    best, n = 0.0, len(q)
    step = max(1, n // 4)
    for i in range(0, max(1, len(t) - n + 1), step):
        best = max(best, SequenceMatcher(None, q, t[i:i + n]).ratio())
    return round(best, 3)


def find_quote(quote, chunks, threshold=0.85):
    """Return [(chunk_id, score)] of chunks that contain the quote (score >= threshold), best first.

    `chunks` is a list of dicts with "child_id" and "text".
    """
    hits = [(c["child_id"], quote_score(quote, c["text"])) for c in chunks]
    return sorted([h for h in hits if h[1] >= threshold], key=lambda h: -h[1])
