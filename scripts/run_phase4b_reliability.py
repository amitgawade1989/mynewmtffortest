"""Phase 4b - reliability evaluation (framework R1-R4, Document Comparison column).

Reads: the variant run (outputs/runs/<run_id>) and the clean baseline run that has been evaluated
by Phase 4 (outputs/runs/<baseline>/eval/per_question.jsonl). Nothing else.

  R1 Semantic consistency   paraphrase answers: pairwise cosine >= tau; expected-match rate vs ground truth
  R1 Repeatability (O)      identical runs: exact-match rate to the modal answer; similarity std
  R2 Typo / noise           accuracy(clean) - accuracy(perturbed) per perturbation; flag > drop threshold
  R2 Adversarial            accuracy delta + LLM judge Held / Partially Shifted / Fully Capitulated
  R2 Long context           accuracy heatmap (length x position), slope, lost-in-the-middle gap
  R3 Format                 regex / structural validators per rule -> compliance rate
  R3 Constraint             keyword pre-filter for hard rules, LLM judge for soft rules -> compliance rate
  R3 Multi-step             4 deterministic sub-requirements + 1 judged -> requirement coverage
  R4 Ambiguity              clarification regex, else LLM judge -> appropriate-response rate
  R4 Insufficient evidence  abstention regex, else LLM judge -> correct-abstention rate (+ false-abstention
                            rate on the clean golden answers)
Accuracy = P1 semantic correctness (cosine pre-filter + LLM judge), same method as Phase 4.

Usage: python scripts/run_phase4b_reliability.py --run-id <variant run> --baseline-run-id <clean run>
"""

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import llm as llm_utils
from src.common.runner import run_resumable
from src.common.utils import load_config, read_json, read_jsonl, resolve_path, write_json, write_jsonl
from src.evaluation import performance as pf
from src.evaluation import retrieval_metrics as rm
from src.reliability import evaluate as ev

REQUIRED = {"P1_semantic": ["verdict"], "R2_adversarial": ["verdict"], "R3_constraint": ["pass"],
            "R3_multi_step": ["pass"], "R4_insufficient_evidence": ["verdict"], "R4_ambiguity": ["verdict"]}


def main():
    ap = argparse.ArgumentParser(description="Phase 4b: reliability evaluation")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--baseline-run-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    relcfg, ecfg, rcfg = config["reliability"], config["evaluation"], config["rag"]
    models = llm_utils.model_settings(config)
    doc_ids = [config["alignment"]["doc_a"], config["alignment"]["doc_b"]]
    labels = {d["id"]: d.get("label", d["id"]) for d in config["documents"]}
    runs = resolve_path(rcfg["runs_dir"])
    run_dir, base_dir = runs / args.run_id, runs / args.baseline_run_id
    out_dir = run_dir / "eval_reliability"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {v["qid"]: v for v in read_jsonl(Path(read_json(run_dir / "manifest.json")["settings"]["questions_file"]))}
    gens = {g["qid"]: g for g in read_jsonl(run_dir / "generation.jsonl")}
    rets = {r["qid"]: r for r in read_jsonl(run_dir / "retrieval.jsonl")}
    base_eval = {r["qid"]: r for r in read_jsonl(base_dir / "eval" / "per_question.jsonl")}
    base_gens = {g["qid"]: g for g in read_jsonl(base_dir / "generation.jsonl")}
    golden = {q["qid"]: q for q in read_jsonl(Path(read_json(base_dir / "manifest.json")["settings"]["questions_file"]))}
    if not base_eval:
        sys.exit(f"Baseline {args.baseline_run_id} has no Phase 4 evaluation - run run_phase4_evaluate.py on it first.")
    clean = {q: r["P1_semantic"]["score"] for q, r in base_eval.items()}
    vs = [v for q, v in variants.items() if q in gens]
    print(f"Variants: {len(variants)} | answered: {len(vs)} | baseline evaluated: {len(base_eval)}")

    # embeddings (answers, ground truths) ------------------------------------------------
    texts = {gens[v["qid"]]["answer"] for v in vs} | {v["ground_truth"] for v in vs if v.get("ground_truth")}
    texts |= {base_gens[v["parent_qid"]]["answer"] for v in vs if v.get("parent_qid") in base_gens}
    texts = sorted(t for t in texts if t and t.strip())
    processed = resolve_path(config["paths"]["processed_dir"])
    vec = dict(zip(texts, llm_utils.embed_texts(llm_utils.get_embedder(models["embeddings"]), texts,
                                                processed / "cache" / "embeddings.jsonl")))

    # per-variant deterministic parts + judge tasks ---------------------------------------
    records, tasks = {}, []
    for v in vs:
        g = gens[v["qid"]]
        rec = {"qid": v["qid"], "parent_qid": v.get("parent_qid"), "test": v["test"], "variant": v["variant"],
               "correctness": None}
        if v.get("ground_truth") and v["test"] in ("R1_semantic", "R1_repeatability", "R2_typo_noise",
                                                   "R2_adversarial", "R2_long_context"):
            sim = pf.cos(vec[g["answer"]], vec[v["ground_truth"]]) if g["answer"].strip() else 0.0
            pre = pf.prefilter(sim, ecfg["semantic_thresholds"])
            rec["correctness"] = pre
            rec["semantic_similarity"] = round(sim, 4)
            if pre is None:
                tasks.append(pf.semantic_task(v, g))
            rec["relation_em"] = float((g.get("relation") or "") == v["question_type"])
        if v.get("evidence") and rets[v["qid"]].get("mode", "retrieval") == "retrieval":
            rec["retrieval_child_recall@5"] = rm.question_metrics(v, rets[v["qid"]], doc_ids, [5]).get("combined.child_recall@5")
        if v["test"] == "R2_adversarial":
            rec["technique"] = v["technique"]
            rec["cited_injected_passage"] = any(c["status"] == "injected_false_passage"
                                                for cl in g.get("claims") or [] for c in cl["citations"])
            tasks.append(ev.adversarial_task(v, g, (base_gens.get(v["parent_qid"]) or {}).get("answer")))
        if v["test"] == "R2_long_context":
            rec["long_context"] = v["long_context"]
        if v["test"] == "R3_format":
            rec.update({"rule": v["rule"], "pass": ev.format_ok(v, g, labels)})
        if v["test"] == "R3_constraint":
            rec["rule"] = v["rule"]
            rec["pass"] = ev.constraint_prefilter(v, g)
            if rec["pass"] is None:
                tasks.append(ev.constraint_task(v, g))
        if v["test"] == "R3_multi_step":
            checks, parsed = ev.multistep_deterministic(g, labels)
            rec["checks"] = checks
            if parsed:
                tasks.append(ev.multistep_task(v, g))
            else:
                rec["checks"]["part3_consistent_implication"] = False
        if v["test"] == "R4_insufficient_evidence":
            rec["verdict"] = "Correct Abstention" if ev.abstained(g, doc_ids) else None
            rec["route"] = "auto" if rec["verdict"] else "judge"
            if rec["verdict"] is None:
                tasks.append(ev.abstention_task(v, g))
        if v["test"] == "R4_ambiguity":
            rec["verdict"] = "Clarification Requested" if ev.CLARIFY.search(g["answer"] or "") else None
            rec["route"] = "auto" if rec["verdict"] else "judge"
            if rec["verdict"] is None:
                tasks.append(ev.ambiguity_task(v, g))
        records[v["qid"]] = rec

    # LLM judge -----------------------------------------------------------------------------
    print(f"LLM judge ({models['judge']}): {len(tasks)} calls")
    judge = llm_utils.get_chat(models["judge"], models["temperature"])

    def run_task(t):
        res = llm_utils.call_json(judge, t["prompt"], required_keys=REQUIRED[t["metric"]])
        return {"metric": t["metric"], "qid": t["qid"], "data": res["data"], "raw": res["raw"], "usage": res["usage"]}

    try:
        for t in tasks:                                   # a different judge model never reuses old verdicts
            t["key"] = f"{models['judge']}|{t['key']}"
        done, failed = run_resumable(tasks, lambda t: t["key"], run_task, out_dir / "judge.jsonl",
                                     out_dir / "failed_judge.jsonl", args.retry_failed, "judge calls",
                                     workers=ecfg.get("workers", 4))
    except llm_utils.QuotaExhausted:
        sys.exit("Stopped: quota exhausted. Resume later with the same command.")
    by_key = {d["key"]: d["data"] for d in done}
    for t in tasks:
        rec, d = records[t["qid"]], by_key.get(t["key"])
        if d is None:
            rec["judge_failed"] = True
        elif t["metric"] == "P1_semantic":
            rec["correctness"] = pf.parse_semantic(d)["score"]
            rec["semantic_verdict"] = d.get("verdict")
        elif t["metric"] in ("R3_constraint",):
            rec["pass"] = bool(d.get("pass"))
            rec["judge_reason"] = d.get("reason")
        elif t["metric"] == "R3_multi_step":
            rec["checks"]["part3_consistent_implication"] = bool(d.get("pass"))
        else:
            rec["verdict"] = d.get("verdict")
            rec["judge_reason"] = d.get("reason")

    recs = list(records.values())
    write_jsonl(out_dir / "per_variant.jsonl", recs)
    summary = summarize(recs, gens, base_gens, golden, clean, vec, relcfg, doc_ids, models, args, len(failed))
    write_json(out_dir / "summary.json", summary)
    print_summary(summary)
    print(f"\nSaved to {out_dir}")


def summarize(recs, gens, base_gens, golden, clean, vec, relcfg, doc_ids, models, args, n_failed):
    """Framework formulas per sub-metric + a 0-1 headline score with pass/fail."""
    by_test = lambda t: [r for r in recs if r["test"] == t]
    thresholds = relcfg.get("pass_thresholds", {})
    flag = relcfg.get("accuracy_drop_flag", 0.05)
    detail, headline = {}, {}

    rows = by_test("R1_semantic")
    if rows:
        groups = {}
        for r in rows:
            groups.setdefault(r["parent_qid"], [base_gens[r["parent_qid"]]["answer"]] if r["parent_qid"] in base_gens else [])
            groups[r["parent_qid"]].append(gens[r["qid"]]["answer"])
        detail["R1_semantic"] = {"consistency_rate": ev.semantic_consistency(groups, vec, relcfg["consistency_threshold"]),
                                 "expected_match_rate": ev.mean_or_none(None if r["correctness"] is None else float(r["correctness"] == 1.0) for r in rows),
                                 "mean_correctness": ev.mean_or_none(r["correctness"] for r in rows),
                                 "relation_agreement": ev.mean_or_none(r.get("relation_em") for r in rows),
                                 "n_variants": len(rows)}
        headline["R1_semantic_consistency"] = detail["R1_semantic"]["expected_match_rate"]

    rows = by_test("R1_repeatability")
    if rows:
        groups = {}
        for r in rows:
            groups.setdefault(r["parent_qid"], []).append(gens[r["qid"]]["answer"])
        detail["R1_repeatability"] = ev.repeatability(groups, vec)
        detail["R1_repeatability"]["mean_correctness"] = ev.mean_or_none(r["correctness"] for r in rows)
        headline["R1_repeatability"] = detail["R1_repeatability"]["exact_match_rate"]

    rows = by_test("R2_typo_noise")
    if rows:
        per = ev.accuracy_delta(rows, clean, "variant")
        overall = ev.accuracy_delta([dict(r, all="all") for r in rows], clean, "all").get("all", {})
        detail["R2_typo_noise"] = {"by_perturbation": per, "overall": overall,
                                   "fragile_perturbations": [k for k, d in per.items() if d["delta"] > flag],
                                   "retrieval_child_recall@5": ev.mean_or_none(r.get("retrieval_child_recall@5") for r in rows)}
        headline["R2_typo_noise"] = round(1 - max(0.0, overall.get("delta", 0)), 4) if overall else None

    rows = by_test("R2_adversarial")
    if rows:
        verdicts = Counter(r.get("verdict") for r in rows)
        detail["R2_adversarial"] = {
            "accuracy_delta_by_technique": ev.accuracy_delta(rows, clean, "technique"),
            "behaviour": dict(verdicts),
            "held_rate_by_technique": {t: ev.mean_or_none(float(r.get("verdict") == "Held") for r in rows if r["technique"] == t)
                                       for t in sorted({r["technique"] for r in rows})},
            "cited_planted_passage_rate": ev.mean_or_none(float(r["cited_injected_passage"]) for r in rows
                                                          if r["technique"] == "adv_injected")}
        headline["R2_adversarial"] = ev.mean_or_none(None if r.get("verdict") is None else float(r["verdict"] == "Held") for r in rows)

    rows = by_test("R2_long_context")
    if rows:
        detail["R2_long_context"] = ev.long_context_summary(rows)
        parents = {r["parent_qid"] for r in rows}
        detail["R2_long_context"]["clean_accuracy_same_questions"] = ev.mean_or_none(clean.get(p) for p in parents)
        headline["R2_long_context"] = detail["R2_long_context"]["accuracy"]

    rows = by_test("R3_format")
    if rows:
        detail["R3_format"] = {"compliance_by_rule": {k: ev.mean_or_none(float(r["pass"]) for r in rows if r["rule"] == k)
                                                      for k in sorted({r["rule"] for r in rows})}}
        headline["R3_format"] = ev.mean_or_none(float(r["pass"]) for r in rows)

    rows = by_test("R3_constraint")
    if rows:
        detail["R3_constraint"] = {"compliance_by_rule": {k: ev.mean_or_none(None if r.get("pass") is None else float(r["pass"]) for r in rows if r["rule"] == k)
                                                          for k in sorted({r["rule"] for r in rows})}}
        headline["R3_constraint"] = ev.mean_or_none(None if r.get("pass") is None else float(r["pass"]) for r in rows)

    rows = by_test("R3_multi_step")
    if rows:
        cov = [sum(r["checks"].values()) / 5 for r in rows if "part3_consistent_implication" in r["checks"]]
        detail["R3_multi_step"] = {"requirement_coverage": ev.mean_or_none(cov),
                                   "by_requirement": {k: ev.mean_or_none(float(r["checks"][k]) for r in rows if k in r["checks"])
                                                      for k in sorted({k for r in rows for k in r["checks"]})}}
        headline["R3_multi_step"] = detail["R3_multi_step"]["requirement_coverage"]

    rows = by_test("R4_ambiguity")
    if rows:
        good = ("Clarification Requested", "Multiple Interpretations Acknowledged")
        detail["R4_ambiguity"] = {"verdicts": dict(Counter(r.get("verdict") for r in rows)),
                                  "routes": dict(Counter(r["route"] for r in rows))}
        headline["R4_ambiguity"] = ev.mean_or_none(None if r.get("verdict") is None else float(r["verdict"] in good) for r in rows)

    rows = by_test("R4_insufficient_evidence")
    if rows:
        control = [ev.over_abstained(golden[q], g) for q, g in base_gens.items() if q in golden and golden[q].get("evidence")]
        detail["R4_insufficient_evidence"] = {"verdicts": dict(Counter(r.get("verdict") for r in rows)),
                                              "routes": dict(Counter(r["route"] for r in rows)),
                                              "false_abstention_rate_on_golden": ev.mean_or_none(float(c) for c in control)}
        headline["R4_insufficient_evidence"] = ev.mean_or_none(None if r.get("verdict") is None else float(r["verdict"] == "Correct Abstention") for r in rows)

    return {"created_at": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id,
            "baseline_run_id": args.baseline_run_id, "models": models, "n_variants": len(recs), "judge_failed": n_failed,
            "headline": {k: {"score": v, "threshold": thresholds.get(k),
                             "passed": None if v is None or k not in thresholds else v >= thresholds[k],
                             "n": sum(r["test"] == k.replace("_consistency", "") for r in recs)}
                         for k, v in headline.items()},
            "detail": detail}


def print_summary(s):
    print(f"\n{'Metric':<28}{'Score':>8}{'Threshold':>11}{'Result':>8}{'n':>5}")
    for k, h in s["headline"].items():
        res = "-" if h["passed"] is None else ("PASS" if h["passed"] else "FAIL")
        sc = f"{h['score']:.3f}" if h["score"] is not None else "-"
        print(f"{k:<28}{sc:>8}{str(h['threshold'] or '-'):>11}{res:>8}{h['n']:>5}")
    for k, d in s["detail"].items():
        print(f"\n{k}: {d}")


if __name__ == "__main__":
    main()
