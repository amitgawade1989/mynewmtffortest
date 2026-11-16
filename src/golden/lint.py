"""Deterministic check that question text carries no metadata or document-identity hints."""

import re
from pathlib import Path

# Always blocked, whatever the documents are. (pattern, case_sensitive)
GENERIC_PATTERNS = [
    (r"\bregion\s*[-#]?\s*\d+\b", False),                         # Region 1
    (r"\b[RD]-?\d+\b", True),                                     # R1, D2, D-1, chunk ids
    (r"\bpage\s*(no\.?|number|#)?\s*\d+", False),                 # page 3, page no 3
    (r"\bp\.\s*\d+\b", False),                                    # p. 3
    (r"\b(section|clause|article|paragraph|para)\s*\d+(\.\d+)*\b", False),
    (r"\bphase\s*\d+\b", False),
    (r"\bchunk\b", False),
    (r"\b(?i:document|doc|handbook|policy|file)\s*[-#]?\s*(A|B|1|2)\b", True),   # Document A, handbook 2
    (r"\b(document|handbook|policy)\s+(one|two)\b", False),
    (r"\b(first|second|former|latter)\s+(document|handbook|policy|file)\b", False),
    (r"\b(left|right)\s+(document|handbook)\b", False),
    (r"\.(pdf|docx|txt|md)\b", False),
]


def build_banned(config):
    """Compile all banned patterns: generic ones + configured terms + document ids, labels and file names."""
    terms = list(config.get("golden", {}).get("banned_terms") or [])
    for d in config["documents"]:
        terms += [d["id"], d.get("label", ""), Path(d["path"]).stem]
    terms = [t for t in terms if t and t.strip()]
    patterns = [re.compile(p, 0 if case else re.IGNORECASE) for p, case in GENERIC_PATTERNS]
    patterns += [re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.IGNORECASE) for t in terms]
    return patterns


def lint_question(text, banned):
    """Return the list of banned fragments found in the question text (empty list = clean)."""
    hits = []
    for p in banned:
        m = p.search(text or "")
        if m:
            hits.append(m.group(0))
    return sorted(set(hits))
