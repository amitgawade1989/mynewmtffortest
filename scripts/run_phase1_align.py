"""Phase 1 - topic alignment and fact pairs between two documents.

Steps (each LLM step is resumable; results are saved item by item):
  1. facts            fact inventory per parent section            -> facts_raw.jsonl
  2. section matching LLM-confirmed section matches, both ways     -> section_matches.jsonl
  3. topics           connected groups of matched sections         -> topic_map.json
  4. fact pairing     facts grouped + SAME/DIFFERENT/PARTIAL/...   -> pairing_raw.jsonl
  5. absence check    ONE_SIDED facts searched in the other doc    -> absence_checks.jsonl
  6. assemble         final fact_pairs.jsonl + alignment_review.md + alignment_manifest.json
Failures go to failed_<step>.jsonl. Output folder: data/processed/alignment/

Usage:
  python scripts/run_phase1_align.py                 # run / resume
  python scripts/run_phase1_align.py --retry-failed  # re-run only failed items
  python scripts/run_phase1_align.py --fresh         # delete previous Phase 1 output and start over
"""

import argparse
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alignment import steps
from src.common import llm as llm_utils
from src.common.runner import run_resumable
from src.common.utils import load_config, read_json, read_jsonl, resolve_path, write_json, write_jsonl


def phase0_fingerprint(processed_dir, doc_ids):
    """Hashes of all Phase 0 chunks; if these change, Phase 1 output is stale."""
    return {d: [c["text_hash"] for c in read_jsonl(processed_dir / d / "children.jsonl")] for d in doc_ids}


def main():
    ap = argparse.ArgumentParser(description="Phase 1: topic alignment and fact pairs")
    ap.add_argument("--config", default=None)
    ap.add_argument("--fresh", action="store_true", help="delete previous Phase 1 output first")
    ap.add_argument("--retry-failed", action="store_true", help="re-run only items that failed before")
    args = ap.parse_args()

    config = load_config(args.config)
    acfg = config["alignment"]
    models = llm_utils.model_settings(config)
    processed = resolve_path(config["paths"]["processed_dir"])
    out = processed / "alignment"
    doc_a, doc_b = acfg["doc_a"], acfg["doc_b"]

    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # stop if Phase 0 output or the models changed since the last (partial) Phase 1 run
    fp_path = out / "phase0_fingerprint.json"
    fingerprint = {"chunks": phase0_fingerprint(processed, [doc_a, doc_b]),
                   "models": {k: models.get(k) for k in ("provider", "generator", "embeddings")}}  # models Phase 1 uses
    if fp_path.exists() and read_json(fp_path) != fingerprint:
        sys.exit("Phase 0 chunks or the models changed since the last Phase 1 run. Re-run with --fresh.")
    write_json(fp_path, fingerprint)

    parents_a, children_a = steps.load_chunks(processed, doc_a)
    parents_b, children_b = steps.load_chunks(processed, doc_b)
    if not parents_a or not parents_b:
        sys.exit("Phase 0 output missing - run scripts/run_phase0_ingest.py first.")
    all_parents, all_children = parents_a + parents_b, children_a + children_b
    child_to_parent = {c["child_id"]: c["parent_id"] for c in all_children}

    llm = llm_utils.get_chat(models["generator"], models["temperature"])
    embedder = llm_utils.get_embedder(models["embeddings"])
    cache = processed / "cache" / "embeddings.jsonl"
    rf = args.retry_failed

    try:
        # 1. fact inventory --------------------------------------------------
        print("\n[1/5] Fact inventory per section")
        rows, _ = run_resumable(all_parents, lambda p: p["parent_id"],
                                lambda p: steps.extract_facts(p, all_children, llm, acfg),
                                out / "facts_raw.jsonl", out / "failed_facts.jsonl", rf, "sections")
        facts = [f for r in rows for f in r["facts"]]
        facts_index = {f["fact_id"]: f for f in facts}
        facts_by_doc = {doc_a: [f for f in facts if f["doc_id"] == doc_a],
                        doc_b: [f for f in facts if f["doc_id"] == doc_b]}
        write_jsonl(out / "facts.jsonl", facts)

        # 2. section matching (A -> B, then unmatched B -> A) ----------------
        print("\n[2/5] Section matching")
        va = steps.parent_vectors(parents_a, embedder, cache)
        vb = steps.parent_vectors(parents_b, embedder, cache)
        k, min_sim = acfg.get("section_candidates", 3), acfg.get("section_min_similarity", 0.45)
        cand_ab = steps.section_candidates(parents_a, va, parents_b, vb, k, min_sim)
        rows_ab, _ = run_resumable(parents_a, lambda p: f"{doc_a}->{doc_b}:{p['parent_id']}",
                                   lambda p: steps.match_section(p, cand_ab[p["parent_id"]], llm),
                                   out / "section_matches.jsonl", out / "failed_section_matches.jsonl", rf,
                                   f"{doc_a} sections")
        matched_b = {m["parent_id"] for r in rows_ab for m in steps.primary_matches(r)}
        unmatched_b = [p for p in parents_b if p["parent_id"] not in matched_b]
        cand_ba = steps.section_candidates(parents_b, vb, parents_a, va, k, min_sim)
        rows_ba, _ = run_resumable(unmatched_b, lambda p: f"{doc_b}->{doc_a}:{p['parent_id']}",
                                   lambda p: steps.match_section(p, cand_ba[p["parent_id"]], llm),
                                   out / "section_matches.jsonl", out / "failed_section_matches.jsonl",
                                   rf, f"unmatched {doc_b} sections")
        rows_all = rows_ab + rows_ba

        # 3. topics ------------------------------------------------------------
        print("\n[3/5] Topic groups")
        topics = steps.build_topics(parents_a, parents_b, rows_all, doc_a, doc_b)
        write_json(out / "topic_map.json", {"doc_a": doc_a, "doc_b": doc_b, "topics": topics})
        for t in topics:
            print(f"   {t['topic_id']} [{t['kind']}] {t['name']}: {t['parents'][doc_a]} <-> {t['parents'][doc_b]}")

        # 4. fact pairing ------------------------------------------------------
        print("\n[4/5] Fact pairing per topic")
        pair_rows, _ = run_resumable(topics, steps.topic_key,
                                     lambda t: steps.pair_topic_facts(t, facts_by_doc, llm, doc_a, doc_b),
                                     out / "pairing_raw.jsonl", out / "failed_pairing.jsonl", rf, "topics")
        rows_by_topic = {r["topic_id"]: r for r in pair_rows}
        pairs = [p for t in topics if t["topic_id"] in rows_by_topic
                 for p in steps.build_pairs(t, rows_by_topic[t["topic_id"]], facts_index, doc_a, doc_b)]

        # 5. absence check for ONE_SIDED pairs -------------------------------
        print("\n[5/5] Absence check for one-sided facts")
        vec_children = {doc_a: steps.embed_texts(embedder, [c["text"] for c in children_a], cache),
                        doc_b: steps.embed_texts(embedder, [c["text"] for c in children_b], cache)}
        kids = {doc_a: children_a, doc_b: children_b}
        one_sided = [p for p in pairs if p["relation"] == "ONE_SIDED"]

        def run_check(p):
            present = doc_a if p["sides"][doc_a]["fact_ids"] else doc_b
            other = doc_b if present == doc_a else doc_a
            return steps.check_absence(p, present, other, kids[other], vec_children[other], embedder, cache, llm, acfg)

        checks, _ = run_resumable(one_sided, steps.pair_key, run_check,
                                  out / "absence_checks.jsonl", out / "failed_absence_checks.jsonl", rf, "one-sided facts")
        checks_by_pair = {c["pair_id"]: c for c in checks}
        for p in pairs:
            if p["pair_id"] in checks_by_pair:
                steps.apply_absence_check(p, checks_by_pair[p["pair_id"]], child_to_parent)
            elif p["relation"] == "ONE_SIDED":
                p["flags"].append("absence_check_missing")
    except llm_utils.QuotaExhausted:
        sys.exit("Stopped: daily quota exhausted. Re-run the same command later to resume.")

    # 6. assemble ----------------------------------------------------------
    n_before = len(pairs)
    pairs = steps.drop_heading_pairs(pairs, all_parents)
    if n_before != len(pairs):
        print(f"\nDropped {n_before - len(pairs)} pairs that only compare section headings")
    write_jsonl(out / "fact_pairs.jsonl", pairs)
    failures = {f.stem: len(read_jsonl(f)) for f in out.glob("failed_*.jsonl") if read_jsonl(f)}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": models, "doc_a": doc_a, "doc_b": doc_b,
        "n_facts": {d: len(v) for d, v in facts_by_doc.items()},
        "n_topics": len(topics), "n_pairs": len(pairs),
        "relations": dict(Counter(p["relation"] for p in pairs)),
        "question_types": dict(Counter(p["question_type"] for p in pairs)),
        "pairs_flagged_for_review": sum(bool(p["flags"]) for p in pairs),
        "failures": failures,
        "token_usage": token_usage(out),
    }
    write_json(out / "alignment_manifest.json", manifest)
    write_review(out / "alignment_review.md", topics, pairs, facts, manifest, doc_a, doc_b, config)
    print(f"\nRelations: {manifest['relations']}  | flagged for review: {manifest['pairs_flagged_for_review']}"
          f"  | failures: {failures or 'none'}")
    print(f"Saved to {out}")


def token_usage(out):
    """Total input/output tokens over every saved LLM response in this phase."""
    total = Counter()
    for f in out.glob("*.jsonl"):
        for r in read_jsonl(f):
            u = r.get("usage") or {}
            total["input_tokens"] += u.get("input_tokens", 0)
            total["output_tokens"] += u.get("output_tokens", 0)
            total["calls"] += 1 if u else 0
    return dict(total)


def write_review(path, topics, pairs, facts, manifest, doc_a, doc_b, config):
    """Human-readable review of topics and fact pairs (check this before Phase 2)."""
    label = {d["id"]: d.get("label", d["id"]) for d in config["documents"]}
    cell = lambda s: (s or "").replace("|", "/").replace("\n", " ")
    lines = [f"# Alignment review - {label[doc_a]} ({doc_a}) vs {label[doc_b]} ({doc_b})", "",
             f"- Facts: {manifest['n_facts']} | topics: {manifest['n_topics']} | pairs: {manifest['n_pairs']}",
             f"- Relations: {manifest['relations']}",
             f"- Question types: {manifest['question_types']}",
             f"- Pairs flagged for review: {manifest['pairs_flagged_for_review']} | failures: {manifest['failures'] or 'none'}",
             f"- Facts whose quote was not found in the source: {sum('quote_not_found' in f['flags'] for f in facts)}",
             "", "## Topic map", "", f"| Topic | Name | Kind | {doc_a} sections | {doc_b} sections |", "|---|---|---|---|---|"]
    for t in topics:
        lines.append(f"| {t['topic_id']} | {cell(t['name'])} | {t['kind']} | {'; '.join(t['titles'][doc_a])} "
                     f"| {'; '.join(t['titles'][doc_b])} |")
    for t in topics:
        tp = [p for p in pairs if p["topic_id"] == t["topic_id"]]
        lines += ["", f"## {t['topic_id']} - {cell(t['name'])}", "",
                  f"| Pair | Attribute | {doc_a} | {doc_b} | Relation | Flags |", "|---|---|---|---|---|---|"]
        for p in tp:
            va = "; ".join(p["sides"][doc_a]["values"]) or "—"
            vb = "; ".join(p["sides"][doc_b]["values"]) or "—"
            lines.append(f"| {p['pair_id']} | {cell(p['attribute'])} | {cell(va)} | {cell(vb)} | {p['relation']} "
                         f"| {', '.join(p['flags'])} |")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
