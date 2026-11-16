"""Phase 5 - Excel scorecard + HTML report from the evaluation outputs.

Reads: the framework workbook (rows + Document Comparison applicability), the Phase 4 summary of a
clean run, and the Phase 4b summary of a reliability run. Writes to outputs/reports/<name>/:
  Evaluation_Scorecard.xlsx  framework layout, Result = formula (edit a threshold to re-grade)
  report.html                self-contained report with charts

Usage: python scripts/run_phase5_report.py --run-id <clean run> --reliability-run-id <variant run>
                                           [--safety-run-id <safety run>]
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.utils import load_config, read_json, resolve_path
from src.report import excel, html, scorecard


def main():
    ap = argparse.ArgumentParser(description="Phase 5: scorecard + report")
    ap.add_argument("--run-id", required=True, help="clean golden run evaluated by Phase 4")
    ap.add_argument("--reliability-run-id", default=None, help="variant run evaluated by Phase 4b")
    ap.add_argument("--safety-run-id", default=None, help="safety run evaluated by Phase 4c")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    rep, ecfg = config["report"], config["evaluation"]
    runs = resolve_path(config["rag"]["runs_dir"])
    perf = read_json(runs / args.run_id / "eval" / "summary.json")
    rel = read_json(runs / args.reliability_run_id / "eval_reliability" / "summary.json") if args.reliability_run_id else {"headline": {}, "detail": {}}
    safety = read_json(runs / args.safety_run_id / "eval_safety" / "summary.json") if args.safety_run_id else {"headline": {}, "detail": {}}

    framework = scorecard.read_framework(resolve_path(rep["framework_file"]), rep["capability_column"])
    rows = scorecard.scorecard_rows(framework, perf, rel, ecfg.get("optional_metrics", {}), ecfg.get("pass_thresholds", {}),
                                    safety)

    ks = config["rag"].get("ks", [1, 3, 5, 10])
    retrieval_rows = [["metric"] + [f"@{k}" for k in ks]]
    doc_ids = [config["alignment"]["doc_a"], config["alignment"]["doc_b"]]
    labels = {d["id"]: d.get("label", d["id"]) for d in config["documents"]}
    for scope in doc_ids + ["combined"]:
        for level in ("child", "parent"):
            for m in ("recall", "precision", "hit", "ndcg"):
                name = f"{labels.get(scope, scope)} {level} {m}"
                retrieval_rows.append([name] + [perf["retrieval"].get(f"{scope}.{level}_{m}@{k}", {}).get("mean") for k in ks])
    retrieval_rows.append(["both documents: child hit"] + [perf["retrieval"].get(f"both_sides.child_hit@{k}", {}).get("mean") for k in ks])
    retrieval_rows.append(["combined child MRR", perf["retrieval"].get("combined.child_mrr", {}).get("mean"), None, None, None])

    rel_tables = reliability_tables(rel.get("detail", {}))
    with open(runs / args.run_id / "eval" / "scores.csv", encoding="utf-8") as f:
        per_question = [[to_num(v) for v in r] for r in csv.reader(f)]

    models = perf["models"]
    run_info = {"under_test": models.get("under_test"), "judge": models.get("judge"), "generator": models.get("generator"),
                "embeddings": models.get("embeddings"), "perf_run": args.run_id, "rel_run": args.reliability_run_id or "-",
                "capability": rep["capability_column"], "date": f"{datetime.now():%Y-%m-%d %H:%M}",
                "golden_questions": perf["coverage"]["golden"], "evaluated": perf["coverage"]["evaluated"],
                "reliability_variants": rel.get("n_variants", 0),
                "safety_run": args.safety_run_id or "-", "safety_variants": safety.get("n_variants", 0),
                "judge_failures": (f"performance {perf['coverage']['judge_failed']}, reliability {rel.get('judge_failed', 0)}, "
                                   f"safety {safety.get('judge_failed', 0)}"),
                "documents": ", ".join(f"{d['id']} = {d.get('label')} ({d['path']})" for d in config["documents"]),
                "pricing_note": "P7 cost uses config pricing_usd_per_1m_tokens - verify against current provider pricing."}

    out = resolve_path(rep["out_dir"]) / f"{datetime.now():%Y%m%d_%H%M}_{args.run_id}"
    out.mkdir(parents=True, exist_ok=True)
    excel.build_workbook(out / "Evaluation_Scorecard.xlsx", rows, perf, rel, per_question, run_info, retrieval_rows, rel_tables)
    html.build_html(out / "report.html", rows, perf, rel, run_info, {"ks": ks})
    evaluated = [r for r in rows if r["status"].startswith("Evaluated")]
    print(f"Scorecard rows: {len(rows)} framework sub-metrics | evaluated: {len(evaluated)}")
    print(f"Saved:\n  {out / 'Evaluation_Scorecard.xlsx'}\n  {out / 'report.html'}")


def to_num(v):
    """CSV text -> number where possible (keeps Excel cells numeric)."""
    if v in ("", "None"):
        return None
    try:
        return float(v) if "." in v or "e" in v.lower() else int(v)
    except ValueError:
        return {"True": True, "False": False}.get(v, v)


def reliability_tables(d):
    """Flatten the Phase 4b detail into small tables for the 'Reliability Detail' sheet."""
    t = {}
    if "R1_semantic" in d:
        t["R1 Semantic consistency"] = (["measure", "value"], [[k, v] for k, v in d["R1_semantic"].items()])
    if "R1_repeatability" in d:
        t["R1 Deterministic repeatability"] = (["measure", "value"], [[k, v] for k, v in d["R1_repeatability"].items()])
    if "R2_typo_noise" in d:
        t["R2 Typo / noise - accuracy vs clean"] = (["perturbation", "accuracy", "clean accuracy", "drop", "n"],
                                                    [[k, v["accuracy"], v["clean_accuracy"], v["delta"], v["n"]]
                                                     for k, v in d["R2_typo_noise"]["by_perturbation"].items()])
    if "R2_adversarial" in d:
        a = d["R2_adversarial"]
        t["R2 Adversarial - accuracy and behaviour"] = (
            ["technique", "accuracy", "clean accuracy", "drop", "held rate"],
            [[k, v["accuracy"], v["clean_accuracy"], v["delta"], a["held_rate_by_technique"].get(k)]
             for k, v in a["accuracy_delta_by_technique"].items()])
        t["R2 Adversarial - verdicts"] = (["verdict", "count"], [[k, v] for k, v in a["behaviour"].items()])
    if "R2_long_context" in d:
        lc = d["R2_long_context"]
        lengths = sorted({int(k.split("|")[0]) for k in lc["heatmap"]})
        t["R2 Long context - accuracy (rows = tokens)"] = (["length", "start", "middle", "end"],
                                                           [[L] + [lc["heatmap"].get(f"{L}|{p}") for p in ("start", "middle", "end")] for L in lengths])
        t["R2 Long context - summary"] = (["measure", "value"], [[k, lc[k]] for k in ("slope_per_10x_length", "lost_in_middle_gap",
                                                                                     "accuracy", "clean_accuracy_same_questions")])
    for key, name in (("R3_format", "R3 Format - compliance by rule"), ("R3_constraint", "R3 Constraint - compliance by rule")):
        if key in d:
            t[name] = (["rule", "compliance"], [[k, v] for k, v in d[key]["compliance_by_rule"].items()])
    if "R3_multi_step" in d:
        t["R3 Multi-step - by requirement"] = (["requirement", "met"], [[k, v] for k, v in d["R3_multi_step"]["by_requirement"].items()])
    for key, name in (("R4_ambiguity", "R4 Ambiguity - verdicts"), ("R4_insufficient_evidence", "R4 Insufficient evidence - verdicts")):
        if key in d:
            t[name] = (["verdict", "count"], [[k, v] for k, v in d[key]["verdicts"].items()])
    if "R4_insufficient_evidence" in d:
        t["R4 False abstention on golden set"] = (["measure", "value"],
                                                  [["false_abstention_rate", d["R4_insufficient_evidence"]["false_abstention_rate_on_golden"]]])
    return t


if __name__ == "__main__":
    main()
