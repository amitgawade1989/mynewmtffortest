"""One table of every evaluated query across the three suites (performance, reliability, safety).

Reads the per-query results saved by Phases 4 / 4b / 4c (query_result + retrieval_check on each record)
and joins the question and answer text from the run folders. No scoring happens here.
"""

from collections import Counter, defaultdict
from pathlib import Path

from src.common.utils import read_json, read_jsonl
from src.evaluation.query_results import FRAMEWORK_ROW

SUITES = (("Performance", "eval", "per_question.jsonl"), ("Reliability", "eval_reliability", "per_variant.jsonl"),
          ("Safety", "eval_safety", "per_variant.jsonl"))
STATUS_ORDER = ("PASS", "FAIL", "RETRIEVAL MISS", "N/A")


def load_suite(run_dir, suite, sub, fname):
    """Rows of one suite, or [] when the run was not given / not evaluated."""
    if run_dir is None or not (run_dir / sub / fname).exists():
        return []
    recs = read_jsonl(run_dir / sub / fname)
    if recs and "query_result" not in recs[0]:
        raise SystemExit(f"{run_dir / sub / fname} has no per-query results - re-run its Phase 4 script with "
                         "--summary-only (no LLM calls) to add them.")
    questions = {q["qid"]: q for q in read_jsonl(Path(read_json(run_dir / "manifest.json")["settings"]["questions_file"]))}
    gens = {g["qid"]: g for g in read_jsonl(run_dir / "generation.jsonl")}
    rows = []
    for r in recs:
        q, g, res, rc = questions.get(r["qid"], {}), gens.get(r["qid"], {}), r["query_result"], r.get("retrieval_check") or {}
        test = r.get("test", "golden")
        rows.append({
            "suite": suite, "framework_row": FRAMEWORK_ROW.get(test, test), "test": test,
            "variant": r.get("variant") or r.get("question_type") or "", "qid": r["qid"], "parent_qid": r.get("parent_qid"),
            "status": res["status"], "answer_status": res["answer_status"],
            "retrieval": rc.get("status", "n/a"), "recall@5": rc.get("recall@5"), "retrieval_reason": rc.get("reason", ""),
            "failed_metrics": res["failed_metrics"], "reasons": res["reasons"], "notes": res["notes"],
            "checks": res["checks"], "question": q.get("question", ""), "answer": g.get("answer", ""),
            "difficulty": r.get("difficulty") or q.get("difficulty"), "gap_relation": r.get("relation") if suite == "Safety" else None,
        })
    return rows


def build_rows(runs, perf_run, rel_run, safety_run):
    rows = []
    for (suite, sub, fname), run in zip(SUITES, (perf_run, rel_run, safety_run)):
        rows += load_suite(runs / run if run else None, suite, sub, fname)
    return rows


def status_table(rows):
    """Per suite and framework row: queries, PASS, FAIL, RETRIEVAL MISS, N/A, pass rate of scored queries."""
    groups = defaultdict(Counter)
    for r in rows:
        groups[(r["suite"], r["framework_row"])][r["status"]] += 1
    out = []
    for (suite, fr), c in groups.items():
        scored = c["PASS"] + c["FAIL"]
        out.append({"suite": suite, "framework_row": fr, "queries": sum(c.values()), **{s: c.get(s, 0) for s in STATUS_ORDER},
                    "pass_rate": round(c["PASS"] / scored, 4) if scored else None})
    order = {s: i for i, (s, _, _) in enumerate(SUITES)}
    return sorted(out, key=lambda x: (order[x["suite"]], x["framework_row"]))


def suite_totals(rows):
    out = {}
    for suite, _, _ in SUITES:
        c = Counter(r["status"] for r in rows if r["suite"] == suite)
        if c:
            scored = c["PASS"] + c["FAIL"]
            out[suite] = {"queries": sum(c.values()), **{s: c.get(s, 0) for s in STATUS_ORDER},
                          "pass_rate": round(c["PASS"] / scored, 4) if scored else None}
    return out


def failure_table(rows, examples=4):
    """Which metric failed how often, per framework row, with example query IDs.
    Answer metrics count only scored queries (retrieval misses are one 'retrieval' line, not answer failures)."""
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "RETRIEVAL MISS":
            groups[(r["suite"], r["framework_row"], "retrieval (gold evidence not in top 5)")].append(r["qid"])
            continue
        for m in r["failed_metrics"]:
            groups[(r["suite"], r["framework_row"], m)].append(r["qid"])
    total = Counter((r["suite"], r["framework_row"]) for r in rows)
    scored = Counter((r["suite"], r["framework_row"]) for r in rows if r["status"] != "RETRIEVAL MISS")
    out = []
    for (s, f, m), q in groups.items():
        of = total[(s, f)] if m.startswith("retrieval") else scored[(s, f)]
        out.append({"suite": s, "framework_row": f, "metric": m, "failed": len(q), "of": of,
                    "share": round(len(q) / of, 4) if of else None, "examples": ", ".join(q[:examples])})
    return sorted(out, key=lambda x: (-x["failed"], x["suite"]))


def checks_text(checks):
    """Compact 'metric score ✓/✕' list for one query."""
    parts = []
    for c in checks:
        mark = "–" if c["passed"] is None else ("✓" if c["passed"] else "✕")
        sc = "" if c["score"] is None else f" {c['score']:.2f}"
        parts.append(f"{c['metric']}{sc} {mark}" + ("" if c["counts"] else " (info)"))
    return " | ".join(parts)
