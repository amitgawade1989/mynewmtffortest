"""Phase 2b - build the reliability test suite (framework R1-R4) from the golden dataset.

Deterministic variants: typos/casing/whitespace (R2), repeats (R1), format/constraint/multi-step
instructions (R3), long-context placements (R2).
LLM-generated variants (resumable, saved item by item): paraphrases (R1, lint + similarity checked),
adversarial premise / leading / injected passage (R2), filler text for long context, ambiguous
questions (R4) and unanswerable questions (R4, verified against both full documents).

Output: data/reliability/variants.jsonl (+ filler.json, suite_manifest.json, suite_review.md)
Then:   python scripts/run_phase3_rag.py --questions data/reliability/variants.jsonl --run-id <id>

Usage: python scripts/run_phase2b_reliability_suite.py [--retry-failed]
"""

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alignment.prompts import fill
from src.common import llm as llm_utils
from src.common.runner import run_resumable
from src.common.utils import load_config, read_json, read_jsonl, resolve_path, write_json, write_jsonl
from src.golden.lint import build_banned, lint_question
from src.reliability import build
from src.reliability import prompts as rp

FILLER_TOPICS = ["IT asset management and laptop refresh cycles", "office facilities, meeting rooms and printing",
                 "sustainability reporting and recycling programmes", "software procurement and vendor reviews",
                 "internal knowledge-base and documentation standards", "quarterly town-hall logistics"]


def main():
    ap = argparse.ArgumentParser(description="Phase 2b: reliability test suite")
    ap.add_argument("--config", default=None)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    rcfg = config["reliability"]
    models = llm_utils.model_settings(config)
    seed = rcfg.get("seed", 7)
    out = resolve_path(rcfg["out_dir"])
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    doc_a, doc_b = config["alignment"]["doc_a"], config["alignment"]["doc_b"]
    labels = {d["id"]: d.get("label", d["id"]) for d in config["documents"]}
    banned = build_banned(config)
    processed = resolve_path(config["paths"]["processed_dir"])
    golden = read_jsonl(resolve_path(config["golden"]["out_dir"]) / "golden.jsonl")
    if not golden:
        sys.exit("Golden dataset missing - run Phase 2 first.")

    generator = llm_utils.get_chat(models["generator"], models["temperature"])
    judge = llm_utils.get_chat(models["judge"], models["temperature"])
    embedder = llm_utils.get_embedder(models["embeddings"])
    rf = args.retry_failed
    variants = []
    sample = lambda key: build.stratified_sample(golden, rcfg[key]["n_questions"], seed, key)

    try:
        # R1 semantic: paraphrases ------------------------------------------------------
        print("\n[R1] Paraphrases")
        n_para = rcfg["paraphrase"]["n_paraphrases"]
        rows, _ = run_resumable(sample("paraphrase"), lambda p: f"para:{p['qid']}",
                                lambda p: build.paraphrase_variants(p, generator, n_para, banned),
                                work / "paraphrases.jsonl", work / "failed_paraphrases.jsonl", rf, "questions", workers=4)
        paras = [v for r in rows for v in r["variants"]]
        parents = {q["qid"]: q for q in golden}
        texts = sorted({v["question"] for v in paras} | {parents[v["parent_qid"]]["question"] for v in paras})
        vec = dict(zip(texts, llm_utils.embed_texts(embedder, texts, processed / "cache" / "embeddings.jsonl")))
        from src.evaluation.performance import cos
        kept = [v for v in paras if cos(vec[v["question"]], vec[parents[v["parent_qid"]]["question"]])
                >= rcfg["paraphrase"]["min_similarity"]]
        print(f"   kept {len(kept)} of {len(paras)} paraphrases (similarity >= {rcfg['paraphrase']['min_similarity']})")
        variants += kept

        # R2 adversarial ------------------------------------------------------------------
        print("\n[R2] Adversarial variants")
        rows, _ = run_resumable(sample("adversarial"), lambda p: f"adv:{p['qid']}",
                                lambda p: build.adversarial_variants(p, generator, labels, doc_a, doc_b, banned),
                                work / "adversarial.jsonl", work / "failed_adversarial.jsonl", rf, "questions", workers=4)
        variants += [v for r in rows for v in r["variants"]]

        # filler for long context ---------------------------------------------------------
        print("\n[R2] Long-context filler")
        n_fill = rcfg["long_context"]["filler_paragraphs"]
        batches = [{"i": i, "topics": ", ".join(FILLER_TOPICS[i % len(FILLER_TOPICS):][:3] or FILLER_TOPICS[:3])}
                   for i in range(-(-n_fill // 10))]
        rows, _ = run_resumable(batches, lambda b: f"filler:{b['i']}",
                                lambda b: {"paragraphs": llm_utils.call_json(generator, fill(rp.FILLER, n=10, topics=b["topics"]),
                                                                             required_keys=["paragraphs"])["data"]["paragraphs"]},
                                work / "filler.jsonl", work / "failed_filler.jsonl", rf, "filler batches", workers=4)
        paragraphs = [p for r in sorted(rows, key=lambda r: r["key"]) for p in r["paragraphs"] if isinstance(p, str)]
        write_json(out / "filler.json", {"paragraphs": paragraphs})

        # R4 ambiguity + insufficient evidence ------------------------------------------
        pairs = read_jsonl(processed / "alignment" / "fact_pairs.jsonl")
        subjects = "\n".join(f"- {t}: " + ", ".join(sorted({p['attribute'] for p in pairs if p['topic'] == t}))
                             for t in sorted({p["topic"] for p in pairs}))
        print("\n[R4] Ambiguous questions")
        n_amb = rcfg["ambiguity"]["n_questions"]
        rows, _ = run_resumable([{"id": "ambiguous"}], lambda x: f"amb:{n_amb}",
                                lambda x: llm_utils.call_json(generator, fill(rp.AMBIGUOUS, subjects=subjects, n=n_amb),
                                                              required_keys=["questions"])["data"],
                                work / "ambiguous.jsonl", work / "failed_ambiguous.jsonl", rf, "batch")
        amb = [q for r in rows for q in r["questions"] if isinstance(q, dict) and q.get("question")
               and not lint_question(q["question"], banned)][:n_amb]
        variants += [{"qid": f"R4-AMB-{i:02d}", "parent_qid": None, "test": "R4_ambiguity", "variant": "ambiguous",
                      "question": q["question"], "interpretations": q.get("interpretations", []), "evidence": None}
                     for i, q in enumerate(amb, 1)]

        print("\n[R4] Unanswerable questions (verified against both full documents)")
        n_un = rcfg["insufficient_evidence"]["n_questions"]
        rows, _ = run_resumable([{"id": "unanswerable"}], lambda x: f"unans:{n_un}",
                                lambda x: llm_utils.call_json(generator, fill(rp.UNANSWERABLE, subjects=subjects, n=n_un * 2),
                                                              required_keys=["questions"])["data"],
                                work / "unanswerable.jsonl", work / "failed_unanswerable.jsonl", rf, "batch")
        candidates = [q for r in rows for q in r["questions"] if isinstance(q, str) and not lint_question(q, banned)]
        documents = "\n\n".join(f"=== {labels[d]} ===\n" + "\n\n".join(p["text"] for p in read_jsonl(processed / d / "parents.jsonl"))
                                for d in (doc_a, doc_b))
        checks, _ = run_resumable(candidates, lambda q: f"check:{q}",
                                  lambda q: {"question": q, **llm_utils.call_json(judge, fill(rp.ANSWERABILITY_CHECK, documents=documents, question=q),
                                                                                  required_keys=["answerable"])["data"]},
                                  work / "answerability.jsonl", work / "failed_answerability.jsonl", rf, "candidates", workers=4)
        unans = [c["question"] for c in checks if c.get("answerable") is False][:n_un]
        print(f"   {len(unans)} confirmed unanswerable of {len(candidates)} candidates")
        variants += [{"qid": f"R4-UNANS-{i:02d}", "parent_qid": None, "test": "R4_insufficient_evidence",
                      "variant": "unanswerable", "question": q, "expected_behavior": "abstain", "evidence": None}
                     for i, q in enumerate(unans, 1)]
    except llm_utils.QuotaExhausted:
        sys.exit("Stopped: quota exhausted. Re-run later to resume.")

    # deterministic variants -------------------------------------------------------------
    variants += build.typo_variants(sample("typo"), rcfg["typo"], seed)
    variants += build.repeat_variants(sample("repeatability"), rcfg["repeatability"]["n_runs"])
    variants += build.long_context_variants(sample("long_context"), rcfg["long_context"])
    variants += build.format_variants(sample("format"))
    variants += build.constraint_variants(sample("constraint"))
    variants += build.multi_step_variants(sample("multi_step"))

    write_jsonl(out / "variants.jsonl", variants)
    counts = dict(Counter(v["test"] for v in variants))
    write_json(out / "suite_manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
                                             "models": models, "n_variants": len(variants), "by_test": counts})
    write_review(out / "suite_review.md", variants)
    print(f"\n{len(variants)} variants: {counts}\nSaved to {out / 'variants.jsonl'}")


def write_review(path, variants):
    """A few examples per test for a quick human check."""
    lines = ["# Reliability suite - examples", ""]
    for test in sorted({v["test"] for v in variants}):
        vs = [v for v in variants if v["test"] == test]
        lines += [f"## {test} ({len(vs)} variants)", ""]
        for v in vs[:6]:
            extra = v.get("injected_passage", {}).get("text") if v.get("injected_passage") else ""
            lines.append(f"- `{v['qid']}`: {v['question'][:300]}" + (f"  \n  _planted:_ {extra}" if extra else ""))
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
