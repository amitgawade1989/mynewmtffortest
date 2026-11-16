"""Phase 4 - evaluate a RAG run against the golden dataset (framework: Performance, Document Comparison).

Decoupled: reads only the golden question file and outputs/runs/<run_id>/{retrieval,generation}.jsonl.

  P1 Exact/Objective      deterministic: relation exact match, numeric tolerance match, token F1
  P1 Semantic correctness hybrid: cosine pre-filter, LLM judge for the middle band (polarity / attribution)
  P2 Query relevance      hybrid: cosine pre-filter, LLM judge 1-5 for the middle band
  P3 Completeness         hybrid: key point -> best sentence matching, LLM judge per point, coverage %
  P4 Redundancy           not applicable to Document Comparison (framework)
  P5 Logical coherence    hybrid (optional): NLI contradiction pre-check + LLM judge 1-5
  P5 Linguistic quality   hybrid (optional): readability band + LLM clarity judge
  P6 Claim faithfulness   deterministic: claim -> top-k context chunks -> NLI entailment
  P6 Citation accuracy    deterministic: (cited passages, claim) -> NLI; + cited IDs vs golden evidence IDs
  P7 Latency / tokens     deterministic: TTFT / total latency percentiles vs SLA, tokens, cost
  + retrieval metrics and document-comparison checks
Outputs in outputs/runs/<run_id>/eval/: per_question.jsonl, scores.csv, summary.json, judge.jsonl

Usage: python scripts/run_phase4_evaluate.py --run-id <run_id> [--retry-failed]
"""

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import llm as llm_utils
from src.common.runner import run_resumable
from src.common.utils import load_config, read_json, read_jsonl, resolve_path, write_json, write_jsonl
from src.evaluation import deterministic as det
from src.evaluation import performance as pf
from src.evaluation import retrieval_metrics as rm

METHODS = {
    "P1_exact": "Deterministic", "P1_semantic": "Deterministic + LLM Judge", "P2_relevance": "Deterministic + LLM Judge",
    "P3_completeness": "Deterministic + LLM Judge", "P5_logical_coherence": "Deterministic + LLM Judge",
    "P5_linguistic_quality": "Deterministic + LLM Judge", "P6_claim_faithfulness": "Deterministic",
    "P6_citation_accuracy": "Deterministic", "P7_latency_sla": "Deterministic",
}
REQUIRED = {"P1_semantic": ["verdict"], "P2_relevance": ["score"], "P3_completeness": ["points"],
            "P5_logical_coherence": ["score"], "P5_linguistic_quality": ["score"]}


def main():
    ap = argparse.ArgumentParser(description="Phase 4: evaluation")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    ecfg, rcfg = config["evaluation"], config["rag"]
    models = llm_utils.model_settings(config)
    doc_ids = [config["alignment"]["doc_a"], config["alignment"]["doc_b"]]
    labels = {d["id"]: d.get("label", d["id"]) for d in config["documents"]}
    run_dir = resolve_path(rcfg["runs_dir"]) / args.run_id
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if models["judge"] == models["under_test"]:
        print("WARNING: judge model == model under test (self-grading bias)")

    manifest = read_json(run_dir / "manifest.json")
    golden = {q["qid"]: q for q in read_jsonl(Path(manifest["settings"]["questions_file"]))}
    gens = {g["qid"]: g for g in read_jsonl(run_dir / "generation.jsonl")}
    rets = {r["qid"]: r for r in read_jsonl(run_dir / "retrieval.jsonl")}
    qids = [q for q in golden if q in gens and q in rets and golden[q].get("evidence") is not None]
    print(f"Golden: {len(golden)} | answered: {len(gens)} | evaluated: {len(qids)}")

    processed = resolve_path(config["paths"]["processed_dir"])
    children_index = {c["child_id"]: c for d in doc_ids for c in read_jsonl(processed / d / "children.jsonl")}

    # embeddings for every text the hybrid metrics compare (cached on disk) ---------------
    texts = set()
    for q in qids:
        g, a = golden[q], gens[q]
        texts |= {g["question"], g["ground_truth"], a["answer"], *g["key_points"],
                  *det.split_sentences(a["answer"]), *pf.claim_texts(a)}
        texts |= {c[2] for c in pf.context_chunks(rets[q], children_index)}
    texts = sorted(t for t in texts if t and t.strip())
    print(f"Embedding {len(texts)} texts (cached)...")
    vectors = llm_utils.embed_texts(llm_utils.get_embedder(models["embeddings"]), texts, processed / "cache" / "embeddings.jsonl")
    vec = dict(zip(texts, vectors))

    # deterministic + NLI metrics, and the judge tasks still needed ----------------------
    print("Deterministic and NLI metrics...")
    nli_model, thr = ecfg["nli_model"], ecfg.get("nli_entail_threshold", 0.5)
    opt = ecfg.get("optional_metrics", {})
    records, tasks = {}, []
    for q in qids:
        g, a, r = golden[q], gens[q], rets[q]
        rec = {"qid": q, "question_type": g["question_type"], "reasoning": g.get("reasoning"),
               "difficulty": g.get("difficulty"), "topics": g.get("topics")}
        rec["P1_exact"] = pf.p1_exact(g, a, ecfg.get("numeric_rel_tolerance", 0.01))

        sim = pf.cos(vec[a["answer"]], vec[g["ground_truth"]]) if a["answer"].strip() else 0.0
        pre = pf.prefilter(sim, ecfg["semantic_thresholds"])
        rec["P1_semantic"] = {"similarity": round(sim, 4), "route": route(pre), "score": pre}
        if pre is None:
            tasks.append(pf.semantic_task(g, a))

        sim = pf.cos(vec[g["question"]], vec[a["answer"]]) if a["answer"].strip() else 0.0
        pre = pf.prefilter(sim, ecfg["relevance_thresholds"])
        rec["P2_relevance"] = {"similarity": round(sim, 4), "route": route(pre), "score": pre}
        if pre is None:
            tasks.append(pf.relevance_task(g, a))

        sentences = [s for s in det.split_sentences(a["answer"]) if s in vec]
        pairs = pf.p3_pairs(g["key_points"], sentences, vec)
        rec["P3_completeness"] = {"n_points": len(pairs), "matches": pairs, "score": None}
        if pairs:
            tasks.append(pf.completeness_task(g, a, pairs))

        if opt.get("logical_coherence", True):
            flags, n_pairs = pf.p5_nli_flags(sentences, nli_model, ecfg.get("nli_contradiction_threshold", 0.9), labels)
            rec["P5_logical_coherence"] = {"nli_flags": flags, "nli_pairs": n_pairs, "score": None}
            tasks.append(pf.coherence_task(g, a, flags))
        if opt.get("linguistic_quality", False):
            import textstat
            fre = textstat.flesch_reading_ease(a["answer"])
            lo, hi = ecfg.get("readability_band", [30, 70])
            rec["P5_linguistic_quality"] = {"flesch_reading_ease": fre, "in_band": lo <= fre <= hi, "score": None}
            tasks.append(pf.clarity_task(g, a))

        rec["P6_claim_faithfulness"] = pf.p6_faithfulness(pf.claim_texts(a), pf.context_chunks(r, children_index),
                                                          vec, nli_model, ecfg.get("faithfulness_top_k", 3), thr)
        rec["P6_citation_accuracy"] = pf.p6_citation_accuracy(a, children_index, nli_model, thr)
        rec["citation_ids"] = pf.citation_ids(g, a)
        rec["P7"] = pf.p7(a, ecfg["sla"], ecfg.get("pricing_usd_per_1m_tokens", {}))
        rec["comparison"] = pf.comparison_extras(g, a, doc_ids)
        rec["retrieval"] = rm.question_metrics(g, r, doc_ids, rcfg.get("ks", [1, 3, 5, 10]))
        records[q] = rec

    # LLM judge (parallel, resumable) ----------------------------------------------------
    print(f"\nLLM judge ({models['judge']}): {len(tasks)} calls")
    judge = llm_utils.get_chat(models["judge"], models["temperature"])

    def run_task(t):
        res = llm_utils.call_json(judge, t["prompt"], required_keys=REQUIRED[t["metric"]])
        return {"metric": t["metric"], "qid": t["qid"], "data": res["data"], "raw": res["raw"],
                "usage": res["usage"], "latency_s": res["latency_s"], "model": res["model"]}

    try:
        for t in tasks:                                   # a different judge model never reuses old verdicts
            t["key"] = f"{models['judge']}|{t['key']}"
        done, failed = run_resumable(tasks, lambda t: t["key"], run_task, eval_dir / "judge.jsonl",
                                     eval_dir / "failed_judge.jsonl", args.retry_failed, "judge calls",
                                     workers=ecfg.get("workers", 4))
    except llm_utils.QuotaExhausted:
        sys.exit(f"Stopped: quota exhausted. Resume later with the same command.")
    by_key = {d["key"]: d for d in done}

    for t in tasks:
        rec, d = records[t["qid"]], by_key.get(t["key"])
        m = rec[t["metric"]]
        if d is None:
            m.update({"score": None, "judge_failed": True})
            continue
        if t["metric"] == "P1_semantic":
            m.update(pf.parse_semantic(d["data"]))
        elif t["metric"] == "P3_completeness":
            m.update(pf.parse_completeness(d["data"], m["n_points"]))
        elif t["metric"] == "P5_linguistic_quality":
            clarity = pf.score_1_5(d["data"])
            m.update({"clarity": clarity, "score": None if clarity is None else (float(m["in_band"]) + clarity) / 2})
        else:
            m.update({"score": pf.score_1_5(d["data"]), "judge": d["data"]})

    # outputs ----------------------------------------------------------------------------
    recs = [records[q] for q in qids]
    write_jsonl(eval_dir / "per_question.jsonl", recs)
    write_scores_csv(eval_dir / "scores.csv", recs)
    summary = summarize(recs, ecfg, models, manifest, golden, gens, len(failed))
    write_json(eval_dir / "summary.json", summary)
    print_summary(summary)
    print(f"\nSaved to {eval_dir}")


def route(pre):
    """Name of the hybrid route taken."""
    return "judge" if pre is None else ("auto_pass" if pre == 1.0 else "auto_fail")


def metric_score(rec, metric):
    """Headline 0-1 score of one metric for one question (None if not applicable / failed)."""
    if metric == "P7_latency_sla":
        return float(rec["P7"]["within_sla"])
    return (rec.get(metric) or {}).get("score")


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    return round(mean(vals), 4) if vals else None


def summarize(recs, ecfg, models, manifest, golden, gens, n_judge_failed):
    """Aggregate scores, pass/fail against thresholds, breakdowns and comparison-specific rates."""
    thresholds = ecfg.get("pass_thresholds", {})
    headline = {}
    for metric in METHODS:
        vals = [metric_score(r, metric) for r in recs if metric in r or metric == "P7_latency_sla"]
        if metric not in ("P7_latency_sla",) and not any(metric in r for r in recs):
            continue
        score = mean_or_none(vals)
        headline[metric] = {"score": score, "n": sum(v is not None for v in vals),
                            "n_missing": sum(v is None for v in vals), "method": METHODS[metric],
                            "threshold": thresholds.get(metric),
                            "passed": None if score is None or metric not in thresholds else score >= thresholds[metric]}

    breakdown = {}
    for field in ("question_type", "reasoning", "difficulty"):
        groups = sorted({r[field] for r in recs if r[field]})
        breakdown[field] = {g: {m: mean_or_none(metric_score(r, m) for r in recs if r[field] == g and (m in r or m == "P7_latency_sla"))
                                for m in headline} | {"n": sum(r[field] == g for r in recs)} for g in groups}

    cmp = [r["comparison"] for r in recs]
    cites = sum(c["n_citations"] for c in cmp)
    p7 = [r["P7"] for r in recs]
    sem_judged = [r["P1_semantic"] for r in recs if r["P1_semantic"]["route"] == "judge" and "verdict" in r["P1_semantic"]]
    rate = lambda key: mean_or_none(None if c[key] is None else float(c[key]) for c in cmp)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(), "run_id": manifest["run_id"], "models": models,
        "coverage": {"golden": len(golden), "answered": len(gens), "evaluated": len(recs), "judge_failed": n_judge_failed},
        "headline": headline,
        "sub_metrics": {
            "P1_relation_exact_match": mean_or_none(r["P1_exact"]["relation_em"] for r in recs),
            "P1_numeric_match": mean_or_none(r["P1_exact"]["numeric_match"] for r in recs),
            "P1_token_f1": mean_or_none(r["P1_exact"]["token_f1"] for r in recs),
            "P1_semantic_routes": dict(Counter(r["P1_semantic"]["route"] for r in recs)),
            "P1_polarity_flip_rate": mean_or_none(float(s["polarity_flip"]) for s in sem_judged),
            "P2_relevance_routes": dict(Counter(r["P2_relevance"]["route"] for r in recs)),
            "P5_nli_contradiction_flags": sum(len(r.get("P5_logical_coherence", {}).get("nli_flags", [])) for r in recs),
            "P6_citation_child_precision": mean_or_none(r["citation_ids"]["child_precision"] for r in recs),
            "P6_citation_child_recall": mean_or_none(r["citation_ids"]["child_recall"] for r in recs),
            "P6_citation_parent_precision": mean_or_none(r["citation_ids"]["parent_precision"] for r in recs),
            "P6_citation_parent_recall": mean_or_none(r["citation_ids"]["parent_recall"] for r in recs),
            "P7_ttft_s": det.latency_stats([p["ttft_s"] for p in p7 if p["ttft_s"] is not None]),
            "P7_total_latency_s": det.latency_stats([p["total_s"] for p in p7 if p["total_s"] is not None]),
            "P7_tokens_per_answer": mean_or_none(p["total_tokens"] for p in p7),
            "P7_total_cost_usd": round(sum(p["cost_usd"] or 0 for p in p7), 5) if any(p["cost_usd"] for p in p7) else None,
        },
        "document_comparison": {
            "relation_confusion": {f"{g}->{p}": n for (g, p), n in
                                   Counter((c["gold_relation"], c["pred_relation"]) for c in cmp).items()},
            "equivalence_false_difference_rate": rate("false_difference"),
            "asymmetric_absence_detection_rate": rate("absence_detected"),
            "both_documents_cited_rate": rate("both_docs_cited"),
            "attribution_swap_rate_judged": mean_or_none(float(s["attribution_swap"]) for s in sem_judged),
            "S5_fabricated_citation_rate": round(sum(c["n_fabricated_citations"] for c in cmp) / cites, 4) if cites else None,
            "citations_outside_context_rate": round(sum(c["n_not_in_context_citations"] for c in cmp) / cites, 4) if cites else None,
        },
        "retrieval": rm.aggregate([{k: v for k, v in r["retrieval"].items() if not k.startswith("_")} for r in recs]),
        "breakdown": breakdown,
    }


def write_scores_csv(path, recs):
    """One row per question with every headline score (easy to open in Excel)."""
    cols = ["qid", "question_type", "reasoning", "difficulty"] + list(METHODS) + [
        "relation_em", "numeric_match", "token_f1", "pred_relation", "citation_child_recall", "ttft_s", "total_s"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in recs:
            w.writerow([r["qid"], r["question_type"], r["reasoning"], r["difficulty"]]
                       + [metric_score(r, m) if (m in r or m == "P7_latency_sla") else "" for m in METHODS]
                       + [r["P1_exact"]["relation_em"], r["P1_exact"]["numeric_match"], r["P1_exact"]["token_f1"],
                          r["comparison"]["pred_relation"], r["citation_ids"]["child_recall"],
                          r["P7"]["ttft_s"], r["P7"]["total_s"]])


def print_summary(s):
    """Console view of the headline scores."""
    print(f"\n{'Metric':<24}{'Score':>8}{'Threshold':>11}{'Result':>8}{'n':>5}  Method")
    for m, h in s["headline"].items():
        res = "-" if h["passed"] is None else ("PASS" if h["passed"] else "FAIL")
        sc = f"{h['score']:.3f}" if h["score"] is not None else "-"
        print(f"{m:<24}{sc:>8}{str(h['threshold'] or '-'):>11}{res:>8}{h['n']:>5}  {h['method']}")
    print("\nSub-metrics:", s["sub_metrics"])
    print("Document comparison:", s["document_comparison"])


if __name__ == "__main__":
    main()
