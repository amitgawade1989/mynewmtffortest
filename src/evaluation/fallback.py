"""LLM-judge fallback for deterministic checks that FAILED (config evaluation.fallback_judge).

A deterministic check that passes is final. One that fails is sent to an LLM judge, because the
deterministic methods have known false negatives:
  P6 claim faithfulness   NLI misses paraphrases, table rows, "The X Handbook states ..." framing
  P6 citation accuracy    same NLI weakness on the cited passages (invalid / fabricated IDs stay failed)
  P1 numeric match        regex misses equivalent forms ("two weeks", "25 + 8 = 33")
  R3 format               structure parsers on bullets / tables / markdown (word_limit stays deterministic)
  R3 constraint           keyword rule for "no law names" can fire on innocent words
Checks that are facts (chunk IDs, latency, canary tokens, relation label, exact repeatability) get no fallback.

P6 safeguard: a judge "supported" counts only if its verbatim quote is found in the passages.
Idempotent: the deterministic value is saved once (*_deterministic) and the final value is always
recomputed from it, so re-running never stacks overrides.
"""

import re

from src.alignment.prompts import fill
from src.common.utils import short_hash
from src.evaluation import deterministic as det
from src.reliability import prompts as rp
from src.reliability.rules import FORMAT_RULES, HARD_CONSTRAINT_PATTERNS

CLAIM_SUPPORT = """You check whether a claim is supported by passages from two HR handbooks.

Passages (each starts with [handbook | chunk id]):
<<passages>>

Claim: <<claim>>

The claim is SUPPORTED if the passages state the same thing - paraphrase, table rows and equivalent
numbers count. If the claim names a handbook, the support must come from that handbook's passages.
It is NOT supported if any part of it is missing from, or contradicts, the passages.
An automatic entailment model said "not supported"; check carefully, it is often wrong on tables and paraphrases.
Return JSON only:
{"supported": true, "quote": "<the supporting sentence or table row copied exactly from the passages; empty if not supported>",
 "reason": "<one sentence>"}
"""

NUMERIC = """Question: <<question>>
Reference answer: <<reference>>
Model answer: <<answer>>

A script could not find these reference numbers in the model answer: <<missing>>.
Does the model answer state the SAME values for these points (equivalent forms count, e.g. "two weeks"
for "2 weeks", or "25 days plus 8 bank holidays" for "33 days")? A different value, or a value the
answer does not give, is a mismatch.
Return JSON only: {"numbers_match": true, "reason": "<one sentence>"}
"""

FORMAT = """Formatting instruction given to the model: <<instruction>>
Response: <<answer>>

A script flagged this response as NOT following the instruction. Check carefully whether the response
follows the instruction exactly. Judge the format only, not correctness.
Return JSON only: {"pass": true, "reason": "<one sentence>"}
"""
REQUIRED = {"P6_claim_faithfulness": ["supported"], "P6_citation_accuracy": ["supported"],
            "P1_numeric_match": ["numbers_match"], "R3_format": ["pass"], "R3_constraint": ["pass"]}


def settings(ecfg):
    f = {"enabled": True, "model": None, "format_rules": ["bullets", "table", "no_markdown"],
         "metrics": list(REQUIRED)} | (ecfg.get("fallback_judge") or {})
    return f


def task(metric, qid, idx, prompt, model, **extra):
    return {"metric": metric, "qid": qid, "idx": idx, "prompt": prompt,
            "key": f"{model}|fallback:{metric}:{qid}:{idx}:{short_hash(prompt)}", **extra}


def norm(t):
    return " ".join(re.sub(r"[|*_`#>]", " ", (t or "").lower()).split())


def quote_found(quote, passages):
    """The judge's quote (or every '...'-separated fragment of it) appears in the passages."""
    text = norm(passages)
    parts = [norm(p) for p in re.split(r"\.\.\.|…", quote or "") if len(norm(p)) >= 12]
    return bool(parts) and all(p in text for p in parts)


def passages_text(ids, children_index, labels):
    return "\n".join(f"[{labels.get(children_index[c]['doc_id'], children_index[c]['doc_id'])} | {c}] "
                     f"{children_index[c]['text']}" for c in ids if c in children_index)


# ------------------------------------------------------------------ performance (Phase 4)

def performance_tasks(rec, gold, gen, children_index, labels, fcfg, model):
    """Judge tasks for this question's failed deterministic checks."""
    out, metrics = [], set(fcfg["metrics"])
    if "P6_claim_faithfulness" in metrics:
        for i, c in enumerate(rec["P6_claim_faithfulness"].get("claims") or []):
            if not c["entailed"]:
                p = passages_text(c.get("top_chunks") or [], children_index, labels)
                out.append(task("P6_claim_faithfulness", rec["qid"], i, fill(CLAIM_SUPPORT, passages=p, claim=c["claim"]),
                                model, passages=p))
    if "P6_citation_accuracy" in metrics:
        cites = {c["text"]: [x["child_id"] for x in c.get("citations") or []] for c in gen.get("claims") or []}
        for i, c in enumerate(rec["P6_citation_accuracy"].get("claims") or []):
            if not c["accurate"] and c.get("reason") != "invalid citation" and cites.get(c["claim"]):
                p = passages_text(cites[c["claim"]], children_index, labels)
                out.append(task("P6_citation_accuracy", rec["qid"], i, fill(CLAIM_SUPPORT, passages=p, claim=c["claim"]),
                                model, passages=p))
    ex = rec["P1_exact"]
    nm = ex.get("numeric_match_deterministic", ex.get("numeric_match"))
    if "P1_numeric_match" in metrics and nm is not None and nm < 1:
        missing = missing_numbers(gen.get("answer", ""), gold["ground_truth"])
        out.append(task("P1_numeric_match", rec["qid"], 0, fill(NUMERIC, question=gold["question"], reference=gold["ground_truth"],
                                                                answer=gen.get("answer", ""), missing=", ".join(missing)), model))
    return out


def missing_numbers(prediction, reference, rel_tol=0.01):
    pred = det.extract_numbers(prediction)
    return [f"{r:g}" for r in sorted(set(det.extract_numbers(reference)))
            if not any(abs(p - r) <= max(1e-9, rel_tol * abs(r)) for p in pred)]


def apply_performance(rec, results, model):
    """Write final values from deterministic + judge results (results: (metric, idx) -> task result)."""
    for metric, flag in (("P6_claim_faithfulness", "entailed"), ("P6_citation_accuracy", "accurate")):
        m = rec[metric]
        claims = m.get("claims") or []
        m.setdefault("score_deterministic", m.get("score"))
        judged = overturned = 0
        for i, c in enumerate(claims):
            c["supported"] = c[flag]
            r = results.get((metric, i))
            if c[flag] or r is None:
                continue
            d = r["data"]
            valid = bool(d.get("supported")) and quote_found(d.get("quote"), r["passages"])
            c["fallback"] = {"supported": bool(d.get("supported")), "quote": d.get("quote", ""), "quote_found": valid,
                             "reason": d.get("reason", "")}
            c["supported"] = valid
            judged += 1
            overturned += valid
        if claims:
            m["score"] = sum(c["supported"] for c in claims) / len(claims)
        if judged:
            m["fallback"] = {"judge": model, "judged": judged, "overturned": overturned}
    ex = rec["P1_exact"]
    ex.setdefault("numeric_match_deterministic", ex.get("numeric_match"))
    ex.setdefault("score_deterministic", ex.get("score"))
    nm = ex["numeric_match_deterministic"]
    r = results.get(("P1_numeric_match", 0))
    if r is not None and nm is not None and nm < 1:
        ok = bool(r["data"].get("numbers_match"))
        ex["fallback"] = {"judge": model, "numbers_match": ok, "reason": r["data"].get("reason", "")}
        nm = 1.0 if ok else nm
    ex["numeric_match"] = nm
    parts = [ex["relation_em"]] + ([nm] if nm is not None else [])
    ex["score"] = sum(parts) / len(parts)


# ------------------------------------------------------------------ reliability (Phase 4b)

def format_instruction(v):
    phr = FORMAT_RULES.get(v.get("rule"), [""])
    try:
        i = int(str(v.get("variant", "")).rsplit("_", 1)[1]) - 1
    except (IndexError, ValueError):
        i = 0
    return phr[min(max(i, 0), len(phr) - 1)]


def reliability_tasks(rec, v, gen, fcfg, model):
    det_pass = rec.get("pass_deterministic", rec.get("pass"))
    if rec["test"] == "R3_format" and "R3_format" in fcfg["metrics"] and det_pass is False \
            and rec.get("rule") in fcfg["format_rules"]:
        return [task("R3_format", rec["qid"], 0, fill(FORMAT, instruction=format_instruction(v), answer=gen.get("answer") or ""), model)]
    if rec["test"] == "R3_constraint" and "R3_constraint" in fcfg["metrics"] and det_pass is False \
            and rec.get("rule") in HARD_CONSTRAINT_PATTERNS and (rec.get("route") == "keyword" or not rec.get("judge_reason")):
        return [task("R3_constraint", rec["qid"], 0, fill(rp.CONSTRAINT_JUDGE, instruction=v.get("instruction", ""),
                                                          question=v.get("question", ""), answer=gen.get("answer") or ""), model)]
    return []


def apply_reliability(rec, results, model):
    if rec["test"] not in ("R3_format", "R3_constraint") or rec.get("pass") is None:
        return
    if rec["test"] == "R3_constraint" and rec.get("rule") in HARD_CONSTRAINT_PATTERNS and not rec.get("judge_reason"):
        rec.setdefault("route", "keyword")                   # decided by the keyword rule, not the judge
    if rec["test"] == "R3_constraint" and rec.get("route") != "keyword":
        return
    rec.setdefault("pass_deterministic", rec["pass"])
    r = results.get((rec["test"], 0))
    rec["pass"] = rec["pass_deterministic"]
    if r is not None and rec["pass_deterministic"] is False:
        ok = bool(r["data"].get("pass"))
        rec["fallback"] = {"judge": model, "pass": ok, "reason": r["data"].get("reason", "")}
        rec["pass"] = ok


def run(tasks, model, out_dir, retry_failed=False, workers=4, temperature=0):
    """Run the judge calls (parallel, resumable, cached in fallback_judge.jsonl).
    Returns ({qid: {(metric, idx): {"data", "passages"}}}, number of failed calls)."""
    from src.common import llm as llm_utils
    from src.common.runner import run_resumable

    if not tasks:
        return {}, 0
    judge = llm_utils.get_chat(model, temperature)

    def run_task(t):
        res = llm_utils.call_json(judge, t["prompt"], required_keys=REQUIRED[t["metric"]])
        return {"metric": t["metric"], "qid": t["qid"], "idx": t["idx"], "data": res["data"], "usage": res["usage"]}

    done, failed = run_resumable(tasks, lambda t: t["key"], run_task, out_dir / "fallback_judge.jsonl",
                                 out_dir / "failed_fallback_judge.jsonl", retry_failed, "fallback judge calls", workers=workers)
    by_key = {d["key"]: d for d in done}
    out = {}
    for t in tasks:
        d = by_key.get(t["key"])
        if d is not None:
            out.setdefault(t["qid"], {})[(t["metric"], t["idx"])] = {"data": d["data"], "passages": t.get("passages", "")}
    return out, len(failed)


def stats(recs, metrics):
    """Per metric: deterministic fails reviewed by the judge and how many it overturned."""
    out = {}
    for m in metrics:
        judged = overturned = 0
        for r in recs:
            if m in ("P6_claim_faithfulness", "P6_citation_accuracy") and m in r:
                f = r[m].get("fallback") or {}
                judged += f.get("judged", 0)
                overturned += f.get("overturned", 0)
            elif m == "P1_numeric_match" and "P1_exact" in r and r["P1_exact"].get("fallback"):
                judged += 1
                overturned += r["P1_exact"]["fallback"]["numbers_match"]
            elif r.get("test") == m and r.get("fallback"):
                judged += 1
                overturned += r["fallback"]["pass"]
        if judged:
            out[m] = {"deterministic_fails_judged": judged, "overturned_to_pass": overturned,
                      "confirmed_fail": judged - overturned}
    return out
