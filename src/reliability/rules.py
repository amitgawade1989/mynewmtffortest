"""R3 instruction-following rules: the instruction text (2 phrasings each) and deterministic validators.

Instructions never name a document, so they do not leak document identity into the question.
"""

import re

from src.evaluation.deterministic import split_sentences

FORMAT_RULES = {
    "word_limit": ["Answer in at most 60 words.",
                   "Keep the whole answer to 60 words or fewer."],
    "bullets": ["Answer as exactly 3 bullet points and nothing else.",
                "Give the answer as three bullet points - no more, no fewer, and no other text."],
    "table": ["Present the answer as a markdown table: first column 'Aspect', then one column per handbook "
              "using each handbook's name as the column header.",
              "Format the answer as a markdown table with an 'Aspect' column and one column for each handbook, "
              "headed by the handbook's name."],
    "no_markdown": ["Do not use any markdown formatting (no bullets, headings, bold or tables).",
                    "Reply in plain sentences only, with no markdown of any kind."],
}

CONSTRAINT_RULES = {
    "no_recommendation": "Do not recommend or judge which handbook is better; only describe what each says.",
    "no_law_names": "Do not mention any law, act, regulation or code by name.",
    "on_topic": "Answer only the point asked; do not discuss any other policy area.",
}
HARD_CONSTRAINT_PATTERNS = {   # deterministic pre-filter: a hit is a definite violation
    "no_law_names": re.compile(r"\b(act|regulations?|gdpr|code of practice|directive)\b(\s+\d{4})?", re.I),
}

MULTI_STEP = [
    "Structure the answer in exactly three labelled parts: 'Part 1 - Positions:' what each handbook says; "
    "'Part 2 - Relationship:' whether they are the same, different, or only one addresses it; "
    "'Part 3 - Implication:' one sentence on what this means for an employee.",
    "Reply in three sections titled 'Part 1 - Positions:', 'Part 2 - Relationship:' and 'Part 3 - Implication:'. "
    "In Part 1 give each handbook's position, in Part 2 say whether they are the same, different or one-sided, "
    "and in Part 3 give a single sentence on the practical impact for an employee.",
]

BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+")


def check_format(rule, answer, labels):
    """True if the answer follows the format rule."""
    lines = [l for l in (answer or "").splitlines() if l.strip()]
    if rule == "word_limit":
        return len(answer.split()) <= 60
    if rule == "bullets":
        return sum(bool(BULLET.match(l)) for l in lines) == 3 and all(BULLET.match(l) for l in lines)
    if rule == "table":
        rows = [l for l in lines if l.strip().startswith("|")]
        if len(rows) < 3:
            return False
        header = rows[0].lower()
        return header.count("|") >= 4 and all(lbl.lower() in header for lbl in labels.values())
    if rule == "no_markdown":
        return not any(re.match(r"^\s*(#|[-*•]\s|\d+[.)]\s|\|)", l) for l in lines) and not re.search(r"\*\*|__|`", answer)
    raise ValueError(rule)


def multi_step_parts(answer):
    """Split an answer into its 'Part 1/2/3' sections (None if the labels are missing or out of order)."""
    m = re.search(r"part\s*1\b.*?:(.*?)part\s*2\b.*?:(.*?)part\s*3\b.*?:(.*)", answer or "", re.I | re.S)
    return [p.strip() for p in m.groups()] if m else None


def check_multi_step(answer, labels):
    """Deterministic sub-requirements of the multi-step instruction (the 5th one is judged by an LLM)."""
    parts = multi_step_parts(answer)
    if not parts:
        return {"labelled_parts_in_order": False, "part1_covers_both_documents": False,
                "part2_states_relationship": False, "part3_one_sentence": False}
    return {
        "labelled_parts_in_order": True,
        "part1_covers_both_documents": all(l.lower() in parts[0].lower() for l in labels.values()),
        "part2_states_relationship": bool(re.search(r"\b(same|identical|differ|different|only one|one-sided|"
                                                    r"does not|do not|neither|both|whereas)\b", parts[1], re.I)),
        "part3_one_sentence": len(split_sentences(parts[2])) == 1,
    }
