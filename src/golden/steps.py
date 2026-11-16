"""Phase 2 steps: generate questions from fact pairs, attach evidence, check, critique, revise.

A question record ("candidate") is a plain dict. Evidence is always built from the fact pairs
the question cites (never from IDs the LLM types), so every child/parent ID and quote is real.
"""

import json

from src.alignment.prompts import fill
from src.common.llm import call_json
from src.common.textmatch import find_quote
from src.common.utils import short_hash
from src.golden.lint import lint_question
from src.golden.prompts import CRITIC, GENERATE_CROSS, GENERATE_TOPIC, REVISE

TYPES = ["differential", "equivalence", "asymmetric"]
# Which pair relations can support each question type. PARTIAL (both address it, one with
# materially more detail) is a real difference, so it supports differential questions too.
TYPE_RELATIONS = {"differential": {"DIFFERENT", "PARTIAL"}, "equivalence": {"SAME"},
                  "asymmetric": {"PARTIAL", "ONE_SIDED"}}
FIELDS = ["question", "question_type", "reasoning", "difficulty", "pair_ids", "ground_truth", "short_answer", "key_points"]


# ------------------------------------------------------------------ pair selection

def usable_pairs(pairs, exclude_flags):
    """Pairs that are safe to build questions on.

    Drops pairs with excluded flags, and ONE_SIDED pairs whose absence was not verified.
    """
    out = []
    for p in pairs:
        if set(p["flags"]) & set(exclude_flags):
            continue
        if p["relation"] == "ONE_SIDED" and not (p.get("absence_check") or {}).get("absence_verified"):
            continue
        out.append(p)
    return out


def type_mix(pairs, n):
    """Split n questions over the three types, as evenly as the available pair relations allow."""
    available = [t for t in TYPES if any(p["relation"] in TYPE_RELATIONS[t] for p in pairs)]
    if not available:
        return {}
    mix = {t: n // len(available) for t in available}
    for t in available[: n % len(available)]:
        mix[t] += 1
    return mix


def cross_topic_combos(topic_ids, n_questions, per_call=2):
    """Pick topic pairs for cross-topic questions, spread over the document (2 questions per pair).

    Topic i is paired with the topic half-way through the list, so pairs are far apart
    (e.g. leave + notice) instead of neighbours. Deterministic, so resume keys stay stable.
    """
    n_calls = -(-n_questions // per_call) if n_questions else 0
    k = len(topic_ids)
    if k < 2:
        return []
    combos = []
    for i in range(k):
        a, b = topic_ids[i], topic_ids[(i + k // 2) % k]
        if a != b and (b, a) not in combos:
            combos.append((a, b))
    step = max(1, len(combos) // max(1, n_calls))
    return combos[::step][:n_calls]


def describe_pairs(pairs, labels, doc_a, doc_b, with_quotes=True, group_by_topic=False):
    """Text listing of fact pairs for prompts (optionally under '### Topic Txx: name' headings)."""
    lines, current_topic = [], None
    for p in pairs:
        if group_by_topic and p["topic_id"] != current_topic:
            current_topic = p["topic_id"]
            lines.append(f"\n### Topic {p['topic_id']}: {p['topic']}")
        lines.append(f"[{p['pair_id']}] {p['relation']} | {p['attribute']}")
        for d in (doc_a, doc_b):
            s = p["sides"][d]
            if s["values"]:
                q = f'  (quote: "{s["quotes"][0]}")' if with_quotes and s["quotes"] and s["quotes"][0] else ""
                lines.append(f"    {labels[d]}: {'; '.join(s['values'])}{q}")
            else:
                lines.append(f"    {labels[d]}: NOT ADDRESSED (verified by searching the whole document)")
        if p.get("explanation"):
            lines.append(f"    note: {p['explanation']}")
    return "\n".join(lines)


# ------------------------------------------------------------------ generation

def generate_for_topic(item, llm, gcfg, labels, doc_a, doc_b):
    """One LLM call that writes questions for a topic (or the cross-topic batch)."""
    pairs, n = item["pairs"], item["n"]
    mix = ", ".join(f"{t}: {k}" for t, k in type_mix(pairs, n).items())
    common = {"label_a": labels[doc_a], "label_b": labels[doc_b], "n": n, "mix": mix}
    if item["kind"] == "cross":
        prompt = fill(GENERATE_CROSS, pairs=describe_pairs(pairs, labels, doc_a, doc_b, with_quotes=False,
                                                          group_by_topic=True), **common)
    else:
        prompt = fill(GENERATE_TOPIC, topic=item["topic"], pairs=describe_pairs(pairs, labels, doc_a, doc_b),
                      max_direct=gcfg.get("max_direct_per_topic", 2), **common)
    res = call_json(llm, prompt, required_keys=["questions"])
    questions = [q for q in res["data"]["questions"] if isinstance(q, dict)]
    return {"item_id": item["item_id"], "questions": questions, "raw": res["raw"], "usage": res["usage"],
            "latency_s": res["latency_s"], "model": res["model"]}


def generation_key(item):
    """Resume key: changes when the pairs, the count or the prompt template change."""
    template = GENERATE_CROSS if item["kind"] == "cross" else GENERATE_TOPIC
    parts = [p["pair_id"] for p in item["pairs"]] + [item["n"], template] + ([item["model"]] if item.get("model") else [])
    return f"{item['item_id']}:{short_hash(json.dumps(parts))}"


# ------------------------------------------------------------------ evidence + deterministic checks

def build_evidence(pair_ids, pairs_index, children_index, doc_ids, threshold=0.85):
    """Evidence list (doc, parent, child, quote, pair) and absence list for the cited pairs.

    Each child gets the quote that actually matches it. Children that match none of the side's
    quotes are left out, unless no child matches at all (then all are kept with an empty quote).
    """
    evidence, absent, seen = [], [], set()
    for pid in pair_ids:
        p = pairs_index.get(pid)
        if not p:
            continue
        for d in doc_ids:
            s = p["sides"][d]
            if not s["child_ids"]:
                ac = p.get("absence_check") or {}
                absent.append({"doc_id": d, "pair_id": pid, "attribute": p["attribute"],
                               "searched_parent_ids": ac.get("searched_parent_ids", []),
                               "searched_child_ids": ac.get("searched_child_ids", [])})
                continue
            quotes = [q for q in s["quotes"] if q]
            matched = []
            for cid in s["child_ids"]:
                hits = [(q, find_quote(q, [children_index[cid]], threshold)) for q in quotes]
                hits = [(q, h[0][1]) for q, h in hits if h]
                if hits:
                    matched.append((cid, max(hits, key=lambda x: x[1])[0]))
            if not matched:
                matched = [(cid, "") for cid in s["child_ids"]]
            for cid, quote in matched:
                if (d, cid, pid) in seen:
                    continue
                seen.add((d, cid, pid))
                evidence.append({"doc_id": d, "parent_id": children_index[cid]["parent_id"], "child_id": cid,
                                 "quote": quote, "pair_id": pid})
    return evidence, absent


def make_candidate(q, qid, source, pairs_index, children_index, doc_ids, banned, revision=0):
    """Normalise one generated question, attach evidence and run the deterministic checks."""
    cand = {k: q.get(k) for k in FIELDS}
    cand["pair_ids"] = [p for p in (q.get("pair_ids") or []) if isinstance(p, str)]
    cand["key_points"] = [k for k in (q.get("key_points") or []) if isinstance(k, str)]
    cand["question_type"] = str(cand["question_type"] or "").lower()
    pairs = [pairs_index[p] for p in cand["pair_ids"] if p in pairs_index]
    evidence, absent = build_evidence(cand["pair_ids"], pairs_index, children_index, doc_ids)
    cand.update({
        "qid": qid, "source": source, "revision": revision,
        "topic_ids": sorted({p["topic_id"] for p in pairs}),
        "topics": sorted({p["topic"] for p in pairs}),
        "relations": sorted({p["relation"] for p in pairs}),
        "evidence": evidence, "absent": absent,
        "expected_values": [{"pair_id": p["pair_id"], "attribute": p["attribute"], "relation": p["relation"],
                             **{d: p["sides"][d]["values"] for d in doc_ids}} for p in pairs],
    })
    cand["check_issues"] = deterministic_issues(cand, pairs_index, children_index, doc_ids, banned)
    return cand


def deterministic_issues(cand, pairs_index, children_index, doc_ids, banned):
    """Rule-based problems with a candidate (empty list = passes)."""
    issues = []
    for f in ("question", "ground_truth", "short_answer"):
        if not (cand.get(f) or "").strip():
            issues.append(f"missing field: {f}")
    if len(cand["key_points"]) < 2:
        issues.append("needs at least 2 key_points")
    if cand["question_type"] not in TYPES:
        issues.append(f"unknown question_type '{cand['question_type']}'")
    unknown = [p for p in cand["pair_ids"] if p not in pairs_index]
    if unknown:
        issues.append(f"unknown or excluded pair_ids: {unknown}")
    if not cand["pair_ids"] or unknown == cand["pair_ids"]:
        issues.append("no valid pair_ids cited")
    leaks = lint_question(cand.get("question") or "", banned)
    if leaks:
        issues.append(f"question text reveals metadata / document identity: {leaks}")
    rels = set(cand["relations"])
    if cand["question_type"] in TYPES and not rels & TYPE_RELATIONS[cand["question_type"]]:
        issues.append(f"type '{cand['question_type']}' does not fit the cited pair relations {sorted(rels)}")
    covered = {e["doc_id"] for e in cand["evidence"]} | {a["doc_id"] for a in cand["absent"]}
    if not set(doc_ids) <= covered:
        issues.append("evidence does not cover both documents")
    if not cand["evidence"]:
        issues.append("no positive evidence in either document")
    for e in cand["evidence"]:
        if e["quote"] and not find_quote(e["quote"], [children_index[e["child_id"]]]):
            issues.append(f"quote not found in {e['child_id']}")
    if cand["source"] == "cross" and len(cand["topic_ids"]) < 2:
        issues.append("cross-topic question uses only one topic")
    return issues


# ------------------------------------------------------------------ critic

def evidence_text(cand, children_index, parents_index, labels):
    """Evidence passages (full child text) + absence notes, for the critic."""
    parts, done = [], set()
    for e in cand["evidence"]:
        if e["child_id"] in done:
            continue
        done.add(e["child_id"])
        parts.append(f"[{labels[e['doc_id']]} | {e['child_id']}]\n{children_index[e['child_id']]['text']}")
    for a in cand["absent"]:
        titles = [parents_index[p]["title"] for p in a["searched_parent_ids"] if p in parents_index]
        parts.append(f"[{labels[a['doc_id']]}] does NOT address \"{a['attribute']}\" "
                     f"(verified by searching sections: {', '.join(titles) or 'whole document'})")
    return "\n\n".join(parts)


def critique(cand, llm, children_index, parents_index, labels, doc_a, doc_b):
    """LLM review of one candidate. Returns the critic's scores and issues."""
    prompt = fill(CRITIC, label_a=labels[doc_a], label_b=labels[doc_b], question=cand["question"],
                  qtype=cand["question_type"], ground_truth=cand["ground_truth"],
                  key_points="; ".join(cand["key_points"]),
                  evidence=evidence_text(cand, children_index, parents_index, labels))
    res = call_json(llm, prompt, required_keys=["answerable", "correct", "unambiguous"])
    return {"qid": cand["qid"], "critic": res["data"], "raw": res["raw"], "usage": res["usage"],
            "latency_s": res["latency_s"], "model": res["model"]}


def candidate_key(cand):
    """Resume key for critic / revision calls: changes whenever the question content changes."""
    return f"{cand['qid']}:r{cand['revision']}:{short_hash(json.dumps({k: cand[k] for k in FIELDS}, sort_keys=True))}"


def critic_issues(critic, min_score):
    """Turn critic output into a list of blocking issues (empty list = accepted)."""
    issues = []
    for k in ("answerable", "correct", "unambiguous"):
        try:
            score = float(critic.get(k, 0))
        except (TypeError, ValueError):
            score = 0
        if score < min_score:
            issues.append(f"critic: {k} score {critic.get(k)} < {min_score}")
    rules = {"needs_both_documents": True, "type_matches": True, "leaks_document_identity": False, "too_trivial": False}
    for k, want in rules.items():
        if bool(critic.get(k, want)) != want:
            issues.append(f"critic: {k} = {critic.get(k)}")
    if issues:
        issues += [f"critic note: {i}" for i in critic.get("issues") or []]
    return issues


# ------------------------------------------------------------------ revision

def revise(cand, issues, llm, pairs_for_revision, labels, doc_a, doc_b):
    """Ask the generator to rewrite a rejected question using the review feedback."""
    q = {k: cand[k] for k in FIELDS}
    prompt = fill(REVISE, label_a=labels[doc_a], label_b=labels[doc_b], question=json.dumps(q, ensure_ascii=False),
                  issues="\n".join(f"- {i}" for i in issues), qtype=cand["question_type"],
                  pairs=describe_pairs(pairs_for_revision, labels, doc_a, doc_b))
    res = call_json(llm, prompt, required_keys=["question"])
    return {"qid": cand["qid"], "question": res["data"]["question"], "raw": res["raw"], "usage": res["usage"],
            "latency_s": res["latency_s"], "model": res["model"]}


# ------------------------------------------------------------------ final record

def golden_record(cand, critic):
    """The clean record written to golden.jsonl."""
    return {
        "qid": cand["qid"],
        "question": cand["question"],
        "question_type": cand["question_type"],
        "reasoning": cand["reasoning"],
        "difficulty": cand["difficulty"],                       # generator's intended difficulty
        "critic_difficulty": (critic or {}).get("difficulty"),  # judge's view, with evidence in hand
        "topics": cand["topics"], "topic_ids": cand["topic_ids"],
        "ground_truth": cand["ground_truth"],
        "short_answer": cand["short_answer"],
        "key_points": cand["key_points"],
        "pair_ids": cand["pair_ids"],
        "relations": cand["relations"],
        "expected_values": cand["expected_values"],
        "evidence": cand["evidence"],
        "absent": cand["absent"],
        "source": cand["source"], "revision": cand["revision"],
        "critic_scores": {k: (critic or {}).get(k) for k in ("answerable", "correct", "unambiguous")},
    }
