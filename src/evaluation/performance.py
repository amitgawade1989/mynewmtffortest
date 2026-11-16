"""Performance metrics (framework P1-P7) for one question, plus document-comparison extras.

Each function takes plain dicts and returns plain dicts. Hybrid metrics are split in two:
a deterministic pre-filter that may settle the score, and a judge prompt for the rest.
"""

import json
import re

import numpy as np

from src.alignment.prompts import fill
from src.common.utils import short_hash
from src.evaluation import deterministic as det
from src.evaluation import judge_prompts as jp
from src.evaluation.nli import nli_probs

VERDICT_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def cos(a, b):
    """Cosine similarity of two vectors."""
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


ABSENCE_CLAIM = re.compile(r"\b(does not|doesn't|do not|did not)\s+(mention|address|specify|state|include|"
                           r"provide|cover|refer|say|detail|set out|describe)|\bno (mention|reference)\b|"
                           r"\bnot (addressed|specified|mentioned|stated|covered|included)\b|\bis silent\b", re.I)


def is_absence_claim(text):
    """Claims that something is NOT in a document can't be entailed by any passage; they are scored
    by the absence-detection metric instead of P6 NLI."""
    return bool(ABSENCE_CLAIM.search(text or ""))


def claim_texts(gen):
    """Atomic claims from the model's structured output (LLM-based extraction, framework option 1b)."""
    claims = [c["text"] for c in gen.get("claims") or [] if (c.get("text") or "").strip()]
    return claims or det.split_sentences(gen.get("answer", ""))


def context_chunks(retrieval, children_index):
    """(child_id, doc_id, text) of every chunk the model saw in its context."""
    ids = [c for info in retrieval["context_info"].values() for c in info["child_ids"]]
    return [(c, children_index[c]["doc_id"], children_index[c]["text"]) for c in ids]


# ------------------------------------------------------------------ P1 correctness

def p1_exact(gold, gen, rel_tol):
    """Deterministic correctness: relation exact match, numeric tolerance match, token F1."""
    relation_em = 1.0 if (gen.get("relation") or "").lower() == gold["question_type"] else 0.0
    numeric = det.numeric_match(gen.get("answer", ""), gold["ground_truth"], rel_tol)
    parts = [relation_em] + ([numeric] if numeric is not None else [])
    return {"relation_em": relation_em, "numeric_match": numeric,
            "token_f1": round(det.token_f1(gen.get("answer", ""), gold["ground_truth"]), 4),
            "score": sum(parts) / len(parts)}


def prefilter(similarity, thresholds):
    """Hybrid pre-filter: 1.0 above auto_pass, 0.0 below auto_fail, None = send to the judge."""
    if similarity >= thresholds["auto_pass"]:
        return 1.0
    if similarity <= thresholds["auto_fail"]:
        return 0.0
    return None


# ------------------------------------------------------------------ P3 completeness

def p3_pairs(key_points, sentences, vec):
    """Best-matching answer sentence for each required point (free-text option 2b)."""
    pairs = []
    for kp in key_points:
        if sentences:
            sims = [cos(vec[kp], vec[s]) for s in sentences]
            i = int(np.argmax(sims))
            pairs.append((kp, sentences[i], round(sims[i], 3)))
        else:
            pairs.append((kp, "", 0.0))
    return pairs


# ------------------------------------------------------------------ P5 coherence

def mentioned_docs(sentence, labels):
    """Document ids whose label appears in the sentence (e.g. 'UK Handbook')."""
    s = sentence.lower()
    return {d for d, label in labels.items() if label.lower() in s}


def p5_nli_flags(sentences, nli_model, threshold, labels=None):
    """Sentence pairs the NLI model labels as contradiction above threshold (both directions checked).

    In a comparison, "Doc A says X" vs "Doc B says Y" is not a contradiction, so pairs whose
    sentences each name a single, different document are skipped.
    """
    def about_different_docs(a, b):
        da, db = mentioned_docs(a, labels or {}), mentioned_docs(b, labels or {})
        return len(da) == 1 and len(db) == 1 and da != db

    pairs = [(a, b) for i, a in enumerate(sentences) for j, b in enumerate(sentences)
             if i != j and not about_different_docs(a, b)]
    probs = nli_probs(nli_model, pairs)
    flags = [{"a": a, "b": b, "p": round(p["contradiction"], 3)} for (a, b), p in zip(pairs, probs)
             if p["contradiction"] >= threshold]
    return flags, len(pairs)


# ------------------------------------------------------------------ P6 groundedness

def premise_units(text):
    """Split a chunk into NLI-sized premises: sentences, table rows as 'header: cell' text, and
    adjacent pairs of units. Sentence-level NLI models judge long multi-sentence premises as
    'neutral', so entailment is taken as the max over these units (SummaC-style)."""
    units, header = [], None
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if header is None:
                header = cells
                continue
            units.append("; ".join(f"{h}: {c}" for h, c in zip(header, cells) if c))
        else:
            header = None
            units.extend(det.split_sentences(line) or [line])
    pairs = [f"{a} {b}" for a, b in zip(units, units[1:])]
    return units + pairs or [text]


def strip_attribution(claim, labels, subject="this handbook"):
    """Remove a leading 'The <document> states (that)' and replace remaining document labels with
    a neutral subject, so the claim can be checked against the chunk text itself (which document
    a claim belongs to is checked separately via its citation)."""
    import re
    names = "|".join(re.escape(l) for l in labels.values()) if labels else "x^"
    pattern = (rf"^\s*(the\s+)?({names}|both handbooks?|both documents?|each handbook|the handbook)\s*,?\s*"
               r"(also\s+)?(states?|says|specifies|notes|mentions|indicates|explains|confirms)\s+(that\s+)?")
    out = re.sub(pattern, "", claim, flags=re.IGNORECASE).strip()
    out = out if len(out) > 10 else claim
    for label in (labels or {}).values():                 # "The UK Handbook does not..." -> "This handbook does not..."
        out = re.sub(rf"\b(the\s+)?{re.escape(label)}\b", subject, out, flags=re.IGNORECASE)
    return out[:1].upper() + out[1:]


def entailment(premise_text, claim, nli_model, labels=None):
    """Max entailment probability of the claim over premise units and two neutral phrasings of
    the claim ('this handbook ...' and 'the Company ...', matching how policies are worded)."""
    hyps = {strip_attribution(claim, labels or {}, s) for s in ("this handbook", "the Company")}
    pairs = [(u, h) for u in premise_units(premise_text) for h in hyps]
    return max((p["entailment"] for p in nli_probs(nli_model, pairs)), default=0.0)


def p6_faithfulness(claims, chunks, vec, nli_model, k, threshold, labels=None):
    """Claim faithfulness: each claim vs its top-k most similar context chunks (and the best chunk
    of each document combined, since comparison claims often need both). Entailed if any premise
    entails the claim."""
    n_absence = sum(is_absence_claim(c) for c in claims)
    claims = [c for c in claims if not is_absence_claim(c)]
    if not claims:
        return {"score": None, "n_claims": 0, "n_absence_claims_excluded": n_absence, "claims": []}
    results = []
    for claim in claims:
        ranked = sorted(chunks, key=lambda c: -cos(vec[claim], vec[c[2]]))[:k]
        best = max((entailment(c[2], claim, nli_model, labels) for c in ranked), default=0.0)
        results.append({"claim": claim, "entailment": round(best, 3), "entailed": best >= threshold,
                        "top_chunks": [c[0] for c in ranked]})
    return {"score": sum(r["entailed"] for r in results) / len(results), "n_claims": len(results),
            "n_absence_claims_excluded": n_absence, "claims": results}


def p6_citation_accuracy(gen, children_index, nli_model, threshold, labels=None):
    """Citation accuracy: NLI on (cited passages, claim) for every claim that has citations.
    Citations that are fabricated or not in the context count as inaccurate."""
    rows = []
    for claim in gen.get("claims") or []:
        cites = claim.get("citations") or []
        if not cites or is_absence_claim(claim.get("text")):
            continue
        if any(c["status"] != "ok" for c in cites):
            rows.append({"claim": claim["text"], "accurate": False, "reason": "invalid citation"})
            continue
        premise = "\n".join(children_index[c["child_id"]]["text"] for c in cites)
        p = entailment(premise, claim["text"], nli_model, labels)
        rows.append({"claim": claim["text"], "accurate": p >= threshold, "entailment": round(p, 3)})
    score = sum(r["accurate"] for r in rows) / len(rows) if rows else None
    return {"score": score, "n_cited_claims": len(rows), "claims": rows}


def citation_ids(gold, gen):
    """Cited child / parent IDs vs the golden evidence IDs: precision, recall, F1."""
    gold_children = {e["child_id"] for e in gold["evidence"]}
    gold_parents = {e["parent_id"] for e in gold["evidence"]}
    cp, cr, cf = det.set_prf(gen.get("cited_child_ids", []), gold_children)
    pp, pr, pf = det.set_prf(gen.get("cited_parent_ids", []), gold_parents)
    return {"child_precision": cp, "child_recall": cr, "child_f1": cf,
            "parent_precision": pp, "parent_recall": pr, "parent_f1": pf}


# ------------------------------------------------------------------ P7 efficiency

def p7(gen, sla, pricing):
    """Latency (TTFT / total) vs SLA, tokens and cost for one answer."""
    usage = gen.get("usage") or {}
    return {"ttft_s": gen.get("ttft_s"), "total_s": gen.get("latency_s"),
            "within_sla": (gen.get("ttft_s") or 0) <= sla["ttft_s"] and (gen.get("latency_s") or 0) <= sla["total_s"],
            "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            "cost_usd": det.cost_usd(usage, gen.get("model", ""), pricing)}


# ------------------------------------------------------------------ document comparison extras

def comparison_extras(gold, gen, doc_ids):
    """Comparison-specific checks: relation pair, false difference on equivalence questions,
    absence detection on asymmetric questions, both documents cited, fabricated citations (S5)."""
    pred = (gen.get("relation") or "").lower()
    absent_docs = {a["doc_id"] for a in gold["absent"]}
    nf_docs = {n["doc_id"] for n in gen.get("not_found") or [] if n.get("doc_id")}
    gold_docs = {e["doc_id"] for e in gold["evidence"]}
    cites = [c for cl in gen.get("claims") or [] for c in cl["citations"]]
    return {
        "gold_relation": gold["question_type"], "pred_relation": pred,
        "false_difference": (pred in ("differential", "asymmetric")) if gold["question_type"] == "equivalence" else None,
        "absence_detected": (absent_docs <= nf_docs) if absent_docs else None,
        "both_docs_cited": (set(gen.get("cited_doc_ids", [])) >= gold_docs) if len(gold_docs) == len(doc_ids) else None,
        "n_citations": len(cites),
        "n_fabricated_citations": sum(c["status"] == "fabricated" for c in cites),
        "n_not_in_context_citations": sum(c["status"] == "not_in_context" for c in cites),
    }


# ------------------------------------------------------------------ judge tasks

def judge_task(metric, qid, prompt):
    """A judge call description; the key changes if the prompt changes (safe resume)."""
    return {"metric": metric, "qid": qid, "prompt": prompt, "key": f"{metric}:{qid}:{short_hash(prompt)}"}


def semantic_task(gold, gen):
    return judge_task("P1_semantic", gold["qid"], fill(jp.SEMANTIC, question=gold["question"],
                                                       reference=gold["ground_truth"], answer=gen["answer"]))


def relevance_task(gold, gen):
    return judge_task("P2_relevance", gold["qid"], fill(jp.RELEVANCE, question=gold["question"], answer=gen["answer"]))


def completeness_task(gold, gen, pairs):
    listing = "\n".join(f"{i}. POINT: {kp}\n   BEST SENTENCE (sim {s}): {sent}" for i, (kp, sent, s) in enumerate(pairs, 1))
    return judge_task("P3_completeness", gold["qid"], fill(jp.COMPLETENESS, question=gold["question"],
                                                           answer=gen["answer"], pairs=listing))


def coherence_task(gold, gen, flags):
    listing = "\n".join(f'- "{f["a"]}" vs "{f["b"]}" (p={f["p"]})' for f in flags) or "(none)"
    return judge_task("P5_logical_coherence", gold["qid"], fill(jp.COHERENCE, answer=gen["answer"], flags=listing))


def clarity_task(gold, gen):
    return judge_task("P5_linguistic_quality", gold["qid"], fill(jp.CLARITY, answer=gen["answer"]))


def score_1_5(data):
    """Map a 1-5 judge score to 0-1."""
    try:
        return (min(5, max(1, float(data.get("score")))) - 1) / 4
    except (TypeError, ValueError):
        return None


def parse_semantic(data):
    return {"score": VERDICT_SCORE.get(str(data.get("verdict", "")).lower()),
            "verdict": data.get("verdict"), "polarity_flip": bool(data.get("polarity_flip")),
            "attribution_swap": bool(data.get("attribution_swap")), "reason": data.get("reason", "")}


def parse_completeness(data, n_points):
    points = [p for p in data.get("points") or [] if isinstance(p, dict)]
    ok = sum(bool(p.get("present_and_correct")) for p in points[:n_points])
    return {"score": ok / n_points if n_points else None, "points": points}


def dump(obj):
    return json.dumps(obj, ensure_ascii=False)
