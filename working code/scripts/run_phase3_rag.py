"""Phase 3 - parent-child RAG over two isolated document stores.

For each question:
  1. embed the question once, search EACH document's own store (top k_retrieve children)
  2. save the ranked retrieval (rank, child_id, parent_id, score) -> retrieval.jsonl   [before any LLM call]
  3. expand the top context_k children to parent sections, one context block per document
  4. the model under test answers in JSON with chunk-ID citations       -> generation.jsonl
  5. citations are validated (ok / not_in_context / fabricated) and mapped to parent + document
Every step is saved item by item; re-running with the same --run-id resumes.

Usage:
  python scripts/run_phase3_rag.py --retrieval-only          # retrieval metrics only, no LLM (quality gate)
  python scripts/run_phase3_rag.py                           # full run on the golden set (new run id)
  python scripts/run_phase3_rag.py --run-id <id>             # resume that run
  python scripts/run_phase3_rag.py --questions <file.jsonl>  # any question file with qid + question
  python scripts/run_phase3_rag.py --limit 5                 # quick smoke test
  python scripts/run_phase3_rag.py --rebuild-index           # force re-embedding of the stores
"""

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import llm as llm_utils
from src.common.runner import run_resumable
from src.common.utils import (estimate_tokens, file_sha256, load_config, read_json, read_jsonl,
                              resolve_path, short_hash, write_json)
from src.evaluation import retrieval_metrics as rm
from src.rag import answer as ans
from src.rag import retriever, store
from src.rag.prompts import ANSWER
from src.reliability.long_context import build_long_context


def load_filler(config):
    """Neutral filler paragraphs for the long-context test (made by the Phase 2b script)."""
    path = resolve_path(config["reliability"]["out_dir"]) / "filler.json"
    if not path.exists():
        sys.exit(f"Long-context questions need {path} - run scripts/run_phase2b_reliability_suite.py first.")
    return read_json(path)["paragraphs"]


def main():
    ap = argparse.ArgumentParser(description="Phase 3: parent-child RAG")
    ap.add_argument("--config", default=None)
    ap.add_argument("--questions", default=None, help="question file (default: golden.jsonl)")
    ap.add_argument("--run-id", default=None, help="resume / name a run")
    ap.add_argument("--retrieval-only", action="store_true", help="retrieval + metrics only, no LLM")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    rcfg, acfg = config["rag"], config["alignment"]
    models = llm_utils.model_settings(config)
    processed = resolve_path(config["paths"]["processed_dir"])
    doc_a, doc_b = acfg["doc_a"], acfg["doc_b"]
    doc_ids = [doc_a, doc_b]
    labels = {d["id"]: d.get("label", d["id"]) for d in config["documents"]}
    ks = rcfg.get("ks", [1, 3, 5, 10])

    q_path = Path(args.questions) if args.questions else resolve_path(config["golden"]["out_dir"]) / "golden.jsonl"
    questions = read_jsonl(q_path)[: args.limit] if args.limit else read_jsonl(q_path)
    if not questions:
        sys.exit(f"No questions found in {q_path}")

    # 1. vector stores (one per document) ---------------------------------------
    children_by_doc, children_index, parents_index = store.load_corpus(processed, doc_ids)
    embedder = llm_utils.get_embedder(models["embeddings"])
    cache = processed / "cache" / "embeddings.jsonl"
    embed_fn = lambda texts: llm_utils.embed_texts(embedder, texts, cache)
    store_dir = resolve_path(rcfg["vectorstore_dir"])
    stores = {}
    for d in doc_ids:
        stores[d], rebuilt = store.build_or_load(d, children_by_doc[d], embed_fn, models["embeddings"], store_dir,
                                                 rcfg.get("embed_section_title", True), args.rebuild_index)
        print(f"Store {d}: {len(children_by_doc[d])} children {'(re)built' if rebuilt else 'loaded'} "
              f"[{models['embeddings']}]")

    filler = load_filler(config) if any(q.get("long_context") for q in questions) else []

    def do_retrieval(q):
        if q.get("long_context"):                     # R2 long-context test: retrieval bypassed
            results = {d: [] for d in doc_ids}
            context, info = build_long_context(q, filler, children_by_doc, parents_index, labels, doc_ids)
            mode = "long_context"
        else:
            qv = llm_utils.embed_texts(embedder, [q["question"]], cache)[0]
            results = retriever.retrieve(qv, stores, rcfg.get("k_retrieve", 10))
            context, info = retriever.build_context(results, children_by_doc, parents_index, labels,
                                                    rcfg.get("context_k", 4), rcfg.get("max_parents_per_doc", 3))
            mode = "retrieval"
        if q.get("injected_passage"):                 # R2 adversarial test: planted false passage
            ip = q["injected_passage"]
            context, info = retriever.inject_passage(context, info, ip["doc_id"], ip["text"], labels)
            mode += "+injected"
        return {"qid": q["qid"], "question": q["question"], "mode": mode, "retrieved": results,
                "context_info": info, "context": context, "context_tokens": estimate_tokens(context),
                "isolation_ok": retriever.isolation_ok(results)
                and all(children_index[c]["doc_id"] == d for d, i in info.items() for c in i["child_ids"])}

    # retrieval-only mode: metrics + gate, nothing saved as a run ------------------
    if args.retrieval_only:
        rows = [do_retrieval(q) for q in questions]
        report_retrieval(rows, questions, doc_ids, ks, rcfg, gate=True,
                         out_path=resolve_path(rcfg["runs_dir"]) / "retrieval_checks" /
                         f"{datetime.now():%Y%m%d_%H%M%S}.json")
        return

    # full run ---------------------------------------------------------------------
    run_id = args.run_id or f"{datetime.now():%Y%m%d_%H%M%S}_{models['under_test']}"
    run_dir = resolve_path(rcfg["runs_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    settings = {"models": models, "rag": {k: v for k, v in rcfg.items() if k != "workers"}, "questions_file": str(q_path), "questions_sha256": file_sha256(q_path),
                "answer_prompt_hash": short_hash(ANSWER)}
    if manifest_path.exists() and read_json(manifest_path).get("settings") != settings:
        sys.exit(f"Run {run_id} was made with different settings/questions. Use a new --run-id.")
    write_json(manifest_path, {"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
                               "settings": settings, "status": "running"})
    print(f"Run id: {run_id}  (resume with --run-id {run_id})")

    llm = llm_utils.get_chat(models["under_test"], models["temperature"], seed=models.get("seed"))
    rf = args.retry_failed
    try:
        print("\n[1/2] Retrieval (saved before generation)")
        ret_rows, _ = run_resumable(questions, lambda q: q["qid"], do_retrieval, run_dir / "retrieval.jsonl",
                                    run_dir / "failed_retrieval.jsonl", rf, "questions")
        ret_by_qid = {r["qid"]: r for r in ret_rows}

        def do_generation(q):
            r = ret_by_qid[q["qid"]]
            res = ans.generate_answer(q["question"], r["context"], llm, labels, doc_a, doc_b)
            cit = ans.resolve_citations(res["data"], r["context_info"], children_index, labels)
            return {"qid": q["qid"], "question": q["question"],
                    "relation": res["data"].get("relation"), "answer": res["data"].get("answer", ""),
                    **cit, "isolation_ok": r["isolation_ok"],
                    "raw": res["raw"], "usage": res["usage"], "latency_s": res["latency_s"],
                    "ttft_s": res["ttft_s"], "model": res["model"]}

        print("\n[2/2] Generation by the model under test")
        todo = [q for q in questions if q["qid"] in ret_by_qid]
        gen_rows, failed = run_resumable(todo, lambda q: q["qid"], do_generation, run_dir / "generation.jsonl",
                                         run_dir / "failed_generation.jsonl", rf, "answers",
                                         workers=rcfg.get("workers", 1))
    except llm_utils.QuotaExhausted:
        sys.exit(f"Stopped: quota exhausted. Resume later with --run-id {run_id}")

    # summary ------------------------------------------------------------------------
    print("\nRetrieval metrics (golden evidence vs retrieved IDs):")
    report_retrieval(ret_rows, questions, doc_ids, ks, rcfg, gate=False, out_path=run_dir / "retrieval_metrics.json")
    issues = Counter(i.split(":")[0] for g in gen_rows for i in g["citation_issues"])
    summary = {
        "n_questions": len(questions), "n_retrieved": len(ret_rows), "n_answered": len(gen_rows),
        "n_failed": len(failed), "isolation_violations": sum(not r["isolation_ok"] for r in ret_rows),
        "citation_issues": dict(issues),
        "answers_with_citation_issues": sum(bool(g["citation_issues"]) for g in gen_rows),
        "relation_predicted": dict(Counter(g["relation"] for g in gen_rows)),
        "mean_latency_s": round(sum(g["latency_s"] for g in gen_rows) / max(1, len(gen_rows)), 2),
        "mean_ttft_s": round(sum(g["ttft_s"] for g in gen_rows) / max(1, len(gen_rows)), 2),
        "tokens": {k: sum((g["usage"] or {}).get(k, 0) for g in gen_rows) for k in ("input_tokens", "output_tokens")},
    }
    write_json(manifest_path, {**read_json(manifest_path), "finished_at": datetime.now(timezone.utc).isoformat(),
                               "status": "complete" if not failed and len(gen_rows) == len(questions) else "partial",
                               "summary": summary})
    print(f"\nGeneration: {summary}")
    print(f"Saved to {run_dir}")


def report_retrieval(ret_rows, questions, doc_ids, ks, rcfg, gate, out_path):
    """Compute + print retrieval metrics for questions that have gold evidence; optionally apply the gate."""
    gold = {q["qid"]: q for q in questions if q.get("evidence")}
    per_q = [{"qid": r["qid"], **rm.question_metrics(gold[r["qid"]], r, doc_ids, ks)}
             for r in ret_rows if r["qid"] in gold and r.get("mode", "retrieval").startswith("retrieval")]
    if not per_q:
        print("   (no gold evidence in this question file - retrieval metrics skipped)")
        return
    agg = rm.aggregate([{k: v for k, v in q.items() if k != "qid"} for q in per_q])
    print(rm.summary_table(agg, doc_ids, ks))
    isolation = sum(not r["isolation_ok"] for r in ret_rows)
    print(f"isolation violations: {isolation}")
    weakest = sorted(per_q, key=lambda q: q.get("combined.child_recall@5", 1))[:5]
    print("weakest questions (combined child recall@5): "
          + ", ".join(f"{q['qid']}={q.get('combined.child_recall@5', 0):.2f}" for q in weakest))
    write_json(out_path, {"n_questions": len(per_q), "aggregate": agg, "per_question": per_q,
                          "isolation_violations": isolation})
    print(f"metrics saved to {out_path}")
    if gate:
        g = rcfg.get("retrieval_gate") or {}
        value = agg.get(f"combined.{g.get('metric')}", {}).get("mean")
        if value is not None and value < g.get("min", 0):
            sys.exit(f"RETRIEVAL GATE FAILED: combined {g['metric']} = {value:.3f} < {g['min']}")
        print(f"retrieval gate passed: combined {g.get('metric')} = {value}")


if __name__ == "__main__":
    main()
