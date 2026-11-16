"""R2 long-context test: bury the key evidence inside neutral filler at a given length and position.

Retrieval is bypassed: the context is the gold evidence sections of both documents (plus the
sections searched for an absent side) placed at the start / middle / end of filler text.
"""

from src.common.utils import estimate_tokens
from src.rag.retriever import render_blocks


def evidence_parents(question, doc_ids):
    """Parent sections per document that hold the question's evidence (or were searched for an absence)."""
    parents = {d: [] for d in doc_ids}
    for e in question.get("evidence", []):
        if e["parent_id"] not in parents[e["doc_id"]]:
            parents[e["doc_id"]].append(e["parent_id"])
    for a in question.get("absent", []):
        for pid in a.get("searched_parent_ids", [])[:2]:
            if pid not in parents[a["doc_id"]]:
                parents[a["doc_id"]].append(pid)
    return parents


def filler_text(paragraphs, n_tokens):
    """Cycle through filler paragraphs until about n_tokens of text."""
    out, size, i = [], 0, 0
    while size < n_tokens and paragraphs:
        p = paragraphs[i % len(paragraphs)]
        out.append(p)
        size += estimate_tokens(p)
        i += 1
    return out


def build_long_context(question, filler_paragraphs, children_by_doc, parents_index, labels, doc_ids):
    """Context of ~question['long_context']['length'] tokens with the evidence at the given position."""
    lc = question["long_context"]
    evidence, info = render_blocks(evidence_parents(question, doc_ids), children_by_doc, parents_index, labels)
    filler = filler_text(filler_paragraphs, max(0, lc["length"] - estimate_tokens(evidence)))
    split = {"start": 0, "middle": len(filler) // 2, "end": len(filler)}[lc["position"]]
    wrap = lambda paras: ["=== OTHER COMPANY MATERIAL ===\n" + "\n\n".join(paras)] if paras else []
    parts = wrap(filler[:split]) + [evidence] + wrap(filler[split:])
    return "\n\n".join(parts), info
