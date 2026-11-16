"""Phase 1 steps: fact inventory -> section alignment -> topics -> fact pairs -> absence check.

Relations between the two documents (used by Phase 2 to pick the question type):
    DIFFERENT -> differential     SAME -> equivalence     PARTIAL / ONE_SIDED -> asymmetric
"""

import json
import re

from src.alignment.prompts import ABSENCE_CHECK, FACT_EXTRACTION, FACT_PAIRING, SECTION_MATCH, fill
from src.common.llm import call_json, cosine_matrix, embed_texts
from src.common.textmatch import find_quote, normalize
from src.common.utils import read_jsonl, short_hash

QUESTION_TYPE = {"DIFFERENT": "differential", "SAME": "equivalence", "PARTIAL": "asymmetric", "ONE_SIDED": "asymmetric"}
RELATIONS = set(QUESTION_TYPE)


def load_chunks(processed_dir, doc_id):
    """Read Phase 0 output for one document: (parents, children)."""
    return (read_jsonl(processed_dir / doc_id / "parents.jsonl"),
            read_jsonl(processed_dir / doc_id / "children.jsonl"))


def llm_meta(res):
    """The bits of a call_json result worth saving next to the parsed output."""
    return {"raw": res["raw"], "usage": res["usage"], "latency_s": res["latency_s"], "model": res["model"]}


# ------------------------------------------------------------------ step 1: fact inventory

def extract_facts(parent, children, llm, cfg):
    """Ask the LLM for all comparable facts in one parent section, then verify each quote.

    A fact's child_ids are repaired if its quote is found in a different child of the same parent.
    """
    kids = [c for c in children if c["parent_id"] == parent["parent_id"]]
    chunks = "\n\n".join(f"[{c['child_id']}]\n{c['text']}" for c in kids)
    res = call_json(llm, fill(FACT_EXTRACTION, title=parent["title"], chunks=chunks), required_keys=["facts"])

    facts = []
    for i, f in enumerate(res["data"].get("facts") or [], 1):
        if not isinstance(f, dict) or not f.get("attribute") or not f.get("value"):
            continue
        valid_ids = {c["child_id"] for c in kids}
        child_ids = [cid for cid in (f.get("child_ids") or []) if cid in valid_ids]
        hits = find_quote(f.get("quote", ""), kids, cfg.get("quote_match_threshold", 0.85))
        hit_ids = [h[0] for h in hits]
        flags = []
        if not hits:
            flags.append("quote_not_found")
        elif not set(child_ids) & set(hit_ids):
            child_ids = hit_ids[:1]
            flags.append("child_ids_repaired")
        if not child_ids:
            flags.append("no_valid_child_ids")
        facts.append({
            "fact_id": f"{parent['parent_id']}-F{i:02d}",
            "doc_id": parent["doc_id"],
            "parent_id": parent["parent_id"],
            "child_ids": child_ids,
            "attribute": f["attribute"].strip(),
            "value": f["value"].strip(),
            "category": (f.get("category") or "").strip(),
            "quote": (f.get("quote") or "").strip(),
            "quote_score": hits[0][1] if hits else 0.0,
            "flags": flags,
        })
    return {"doc_id": parent["doc_id"], "parent_id": parent["parent_id"], "facts": facts, **llm_meta(res)}


# ------------------------------------------------------------------ step 2: section alignment

def parent_vectors(parents, embedder, cache_path):
    """Embed each parent as title + start of its text."""
    return embed_texts(embedder, [f"{p['title']}\n{p['text'][:2000]}" for p in parents], cache_path)


def section_candidates(src_parents, src_vecs, tgt_parents, tgt_vecs, k, min_sim):
    """For each source parent, the top-k most similar target parents above min_sim: {pid: [(parent, sim)]}."""
    sims = cosine_matrix(src_vecs, tgt_vecs)
    out = {}
    for i, p in enumerate(src_parents):
        order = sims[i].argsort()[::-1][:k]
        out[p["parent_id"]] = [(tgt_parents[j], round(float(sims[i][j]), 4)) for j in order if sims[i][j] >= min_sim]
    return out


def match_section(parent, candidates, llm):
    """Ask the LLM which candidate sections of the other document overlap with this section."""
    if not candidates:
        return {"parent_id": parent["parent_id"], "topic_name": parent["title"], "matches": [],
                "candidates": [], "note": "no candidate above similarity threshold"}
    cand_text = "\n\n".join(f"[{c['parent_id']}] \"{c['title']}\" (similarity {s}):\n{c['text'][:1500]}"
                            for c, s in candidates)
    res = call_json(llm, fill(SECTION_MATCH, parent_id=parent["parent_id"], title=parent["title"],
                              text=parent["text"][:2500], candidates=cand_text), required_keys=["matches"])
    allowed = {c["parent_id"] for c, _ in candidates}
    matches = [m for m in (res["data"].get("matches") or [])
               if isinstance(m, dict) and m.get("parent_id") in allowed]
    return {"parent_id": parent["parent_id"], "topic_name": res["data"].get("topic_name") or parent["title"],
            "matches": matches, "candidates": [[c["parent_id"], s] for c, s in candidates], **llm_meta(res)}


# ------------------------------------------------------------------ step 3: topics

def primary_matches(row):
    """The matches used for grouping: all 'full' matches, else the single most similar 'partial' one.

    Other partial overlaps are only cross-references; using them for grouping chains unrelated
    sections together (e.g. Intro -> Equality -> Disciplinary -> Notice ...).
    """
    matches = row.get("matches", [])
    full = [m for m in matches if m.get("overlap") == "full"]
    if full or not matches:
        return full
    sim = {pid: s for pid, s in row.get("candidates", [])}
    return [max(matches, key=lambda m: sim.get(m["parent_id"], 0))]


def build_topics(parents_a, parents_b, match_rows, doc_a, doc_b):
    """Merge primary section matches into topic groups (connected components).

    Sections with no match become one-sided topics. Topic IDs follow document order: T01, T02, ...
    """
    group_of = {p["parent_id"]: p["parent_id"] for p in parents_a + parents_b}

    def root(x):
        while group_of[x] != x:
            x = group_of[x]
        return x

    names, related = {}, {}
    for row in match_rows:
        names.setdefault(row["parent_id"], row.get("topic_name"))
        primary = primary_matches(row)
        for m in primary:
            group_of[root(m["parent_id"])] = root(row["parent_id"])
        for m in row.get("matches", []):
            if m not in primary:
                related.setdefault(row["parent_id"], []).append(m["parent_id"])

    groups = {}
    for p in parents_a + parents_b:
        groups.setdefault(root(p["parent_id"]), {doc_a: [], doc_b: []})[p["doc_id"]].append(p)

    order = lambda g: (min([p["order"] for p in g[doc_a]] or [999]), min([p["order"] for p in g[doc_b]] or [999]))
    topics = []
    for i, g in enumerate(sorted(groups.values(), key=order), 1):
        first = (g[doc_a] or g[doc_b])[0]
        kind = "aligned" if g[doc_a] and g[doc_b] else f"only_{doc_a if g[doc_a] else doc_b}"
        topics.append({
            "topic_id": f"T{i:02d}",
            "name": names.get(first["parent_id"]) or first["title"],
            "kind": kind,
            "parents": {doc_a: [p["parent_id"] for p in g[doc_a]], doc_b: [p["parent_id"] for p in g[doc_b]]},
            "titles": {doc_a: [p["title"] for p in g[doc_a]], doc_b: [p["title"] for p in g[doc_b]]},
            "related_parents": sorted({r for p in g[doc_a] + g[doc_b] for r in related.get(p["parent_id"], [])}),
        })
    return topics


def topic_key(topic):
    """Resume key for a topic: changes whenever its sections change, so stale results are not reused."""
    return f"{topic['topic_id']}:{short_hash(json.dumps(topic['parents'], sort_keys=True))}"


def pair_key(pair):
    """Resume key for an absence check: changes whenever the pair's content changes."""
    return f"{pair['pair_id']}:{short_hash(json.dumps([pair['attribute'], pair['sides']], sort_keys=True))}"


# ------------------------------------------------------------------ step 4: fact pairing

def pair_topic_facts(topic, facts_by_doc, llm, doc_a, doc_b):
    """Group and classify the facts of one topic. One-sided topics need no LLM call."""
    fa = [f for f in facts_by_doc[doc_a] if f["parent_id"] in topic["parents"][doc_a]]
    fb = [f for f in facts_by_doc[doc_b] if f["parent_id"] in topic["parents"][doc_b]]
    if not fa or not fb:
        return {"topic_id": topic["topic_id"], "groups": [], "a_ids": [f["fact_id"] for f in fa],
                "b_ids": [f["fact_id"] for f in fb], "note": "one side has no facts - all ONE_SIDED"}

    listing = lambda fs: "\n".join(f"[{f['fact_id']}] {f['attribute']}: {f['value']}" for f in fs)
    res = call_json(llm, fill(FACT_PAIRING, topic=topic["name"], facts_a=listing(fa), facts_b=listing(fb)),
                    required_keys=["pairs"])
    return {"topic_id": topic["topic_id"], "groups": res["data"].get("pairs") or [],
            "a_ids": [f["fact_id"] for f in fa], "b_ids": [f["fact_id"] for f in fb], **llm_meta(res)}


def build_pairs(topic, pairing_row, facts_index, doc_a, doc_b):
    """Turn one topic's LLM grouping into validated pair records.

    Fixes: unknown / wrong-side / duplicate fact IDs are dropped, relation is corrected when it
    contradicts the sides, and facts the LLM forgot are added as ONE_SIDED (flagged for review).
    """
    a_ids, b_ids = set(pairing_row["a_ids"]), set(pairing_row["b_ids"])
    used, pairs = set(), []

    def add(attribute, ids_a, ids_b, relation, explanation, flags):
        flags = list(flags)
        if not ids_a or not ids_b:
            if relation != "ONE_SIDED":
                flags.append(f"relation_corrected_from_{relation}")
            relation = "ONE_SIDED"
        elif relation == "ONE_SIDED" or relation not in RELATIONS:
            flags.append(f"relation_corrected_from_{relation}")
            relation = "PARTIAL"
        pairs.append({
            "pair_id": f"{topic['topic_id']}-{len(pairs) + 1:02d}",
            "topic_id": topic["topic_id"], "topic": topic["name"],
            "attribute": attribute, "relation": relation, "question_type": QUESTION_TYPE[relation],
            "explanation": explanation,
            "sides": {doc_a: side_from_facts(ids_a, facts_index), doc_b: side_from_facts(ids_b, facts_index)},
            "absence_check": None, "flags": flags,
        })

    for g in pairing_row["groups"]:
        if not isinstance(g, dict):
            continue
        ga = [i for i in g.get("a_fact_ids") or [] if i in a_ids and i not in used]
        gb = [i for i in g.get("b_fact_ids") or [] if i in b_ids and i not in used]
        if not ga and not gb:
            continue
        used.update(ga + gb)
        add(g.get("attribute") or facts_index[(ga + gb)[0]]["attribute"], ga, gb,
            str(g.get("relation", "")).upper(), g.get("explanation", ""), [])

    for fid in sorted((a_ids | b_ids) - used):           # facts the LLM did not place
        f = facts_index[fid]
        on_a = fid in a_ids
        flags = ["auto_added_unpaired_fact"] if pairing_row["groups"] else []
        add(f["attribute"], [fid] if on_a else [], [] if on_a else [fid], "ONE_SIDED", "", flags)
    return pairs


def drop_heading_pairs(pairs, parents):
    """Remove pairs whose every value is just a section heading (e.g. "1. Introduction & Scope")."""
    strip_num = lambda s: re.sub(r"^\s*[\dIVXivx.]+\s*", "", normalize(s))
    titles = {strip_num(p["title"]) for p in parents}

    def is_heading_pair(pair):
        values = [v for side in pair["sides"].values() for v in side["values"]]
        return bool(values) and all(strip_num(v) in titles for v in values)

    return [p for p in pairs if not is_heading_pair(p)]


def side_from_facts(fact_ids, facts_index):
    """Evidence for one document side of a pair, built from its facts."""
    facts = [facts_index[i] for i in fact_ids]
    return {
        "fact_ids": fact_ids,
        "values": [f["value"] for f in facts],
        "quotes": [f["quote"] for f in facts],
        "child_ids": sorted({c for f in facts for c in f["child_ids"]}),
        "parent_ids": sorted({f["parent_id"] for f in facts}),
        "source": "fact_inventory",
    }


# ------------------------------------------------------------------ step 5: absence check

def check_absence(pair, present_doc, other_doc, other_children, other_vecs, embedder, cache_path, llm, cfg):
    """Search the whole other document for a ONE_SIDED fact before trusting that it is really absent."""
    side = pair["sides"][present_doc]
    query = f"{pair['attribute']}: {'; '.join(side['values'])}"
    qv = embed_texts(embedder, [query], cache_path)
    sims = cosine_matrix(qv, other_vecs)[0]
    top = [other_children[j] for j in sims.argsort()[::-1][:cfg.get("absence_search_k", 4)]]
    passages = "\n\n".join(f"[{c['child_id']}]\n{c['text']}" for c in top)
    res = call_json(llm, fill(ABSENCE_CHECK, attribute=pair["attribute"], value="; ".join(side["values"]),
                              passages=passages), required_keys=["addressed"])
    d = res["data"]
    valid = {c["child_id"] for c in top}
    child_ids = [c for c in (d.get("child_ids") or []) if c in valid]
    hits = find_quote(d.get("quote", ""), [c for c in top if c["child_id"] in child_ids] or top,
                      cfg.get("quote_match_threshold", 0.85)) if d.get("quote") else []
    return {"pair_id": pair["pair_id"], "present_doc": present_doc, "other_doc": other_doc,
            "searched_child_ids": [c["child_id"] for c in top],
            "addressed": bool(d.get("addressed")) and bool(child_ids),
            "child_ids": child_ids, "value": d.get("value", ""), "quote": d.get("quote", ""),
            "quote_found": bool(hits), "relation": str(d.get("relation", "")).upper(),
            "explanation": d.get("explanation", ""), **llm_meta(res)}


def apply_absence_check(pair, check, child_to_parent):
    """Update a ONE_SIDED pair with its absence-check result (fill the other side if it was found)."""
    pair["absence_check"] = {k: check[k] for k in ("searched_child_ids", "addressed", "explanation")}
    pair["absence_check"]["searched_parent_ids"] = sorted({child_to_parent[c] for c in check["searched_child_ids"]})
    if not check["addressed"]:
        pair["absence_check"]["absence_verified"] = True
        return pair
    relation = check["relation"] if check["relation"] in ("SAME", "DIFFERENT", "PARTIAL") else "PARTIAL"
    pair["sides"][check["other_doc"]] = {
        "fact_ids": [], "values": [check["value"]], "quotes": [check["quote"]],
        "child_ids": check["child_ids"], "parent_ids": sorted({child_to_parent[c] for c in check["child_ids"]}),
        "source": "absence_check",
    }
    pair["relation"], pair["question_type"] = relation, QUESTION_TYPE[relation]
    pair["explanation"] = check["explanation"] or pair["explanation"]
    pair["absence_check"]["absence_verified"] = False
    pair["flags"].append("found_in_other_document_by_absence_check")
    if not check["quote_found"]:
        pair["flags"].append("absence_check_quote_not_found")
    return pair
