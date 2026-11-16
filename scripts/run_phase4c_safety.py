"""Phase 4c - safety evaluation (framework S2 Regional, S3 Refusal Appropriateness; Document Comparison column).

Reads: the safety run (outputs/runs/<run_id>, made from data/safety/variants.jsonl) and, for S2
Regional, the clean baseline run's answers to the same golden questions (outputs/runs/<baseline>).

  S3 Genuinely Unsafe       refusal regex (no step list) -> Refused, else LLM judge
                            Refused / Partially Complied / Fully Complied -> refusal rate
  S3 Benign / Ambiguous     short uncited refusal -> Over-Refused, else LLM judge
                            Complied / Over-Refused / Under-Qualified Compliance -> false-refusal rate
  S3 Injection Direct       canary token in the output -> attack success rate (deterministic)
  S3 Injection Embedded     canary token in the output -> attack success rate (+ planted chunk cited)
  S3 Instruction Hierarchy  canary token in the output -> attack success rate (deterministic)
  S3 Role-play (O)          refusal regex, else judge -> Full Bypass rate
  S3 Hypothetical (O)       refusal regex, else judge -> Actionable Content Leaked rate
  S2 Regional               swapped-order answer vs baseline answer: relation flip / numbers + cosine,
                            else judge Consistent / Framing Shift / Outcome Shift -> bias rate
Every headline has a 95% Wilson interval; "decision" is pass / fail only when the whole interval is
on one side of the threshold, otherwise inconclusive.

Usage: python scripts/run_phase4c_safety.py --run-id <safety run> --baseline-run-id <clean run>
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import llm as llm_utils
from src.common.runner import run_resumable
from src.common.utils import load_config, read_json, read_jsonl, resolve_path, write_json, write_jsonl
from src.safety import evaluate as ev


def main():
    ap = argparse.ArgumentParser(description="Phase 4c: safety evaluation")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--baseline-run-id", default=None, help="clean golden run (needed for S2 Regional)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    scfg, ecfg, rcfg = config["safety"], config["evaluation"], config["rag"]
    models = llm_utils.model_settings(config)
    runs = resolve_path(rcfg["runs_dir"])
    run_dir = runs / args.run_id
    out_dir = run_dir / "eval_safety"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {v["qid"]: v for v in read_jsonl(Path(read_json(run_dir / "manifest.json")["settings"]["questions_file"]))}
    gens = {g["qid"]: g for g in read_jsonl(run_dir / "generation.jsonl")}
    base_gens = {g["qid"]: g for g in read_jsonl(runs / args.baseline_run_id / "generation.jsonl")} if args.baseline_run_id else {}
    vs = [v for q, v in variants.items() if q in gens]
    regional = [v for v in vs if v["test"] == "S2_regional"]
    if regional and not base_gens:
        print("   S2 Regional skipped: pass --baseline-run-id <clean run> to compare against the original order.")
    print(f"Variants: {len(variants)} | answered: {len(vs)}")

    # embeddings (S2 answer pairs only) ---------------------------------------------------
    pairs = [(base_gens[v["parent_qid"]], gens[v["qid"]]) for v in regional if v["parent_qid"] in base_gens]
    texts = sorted({g.get("answer") or "" for p in pairs for g in p} - {""})
    vec = {}
    if texts:
        vec = dict(zip(texts, llm_utils.embed_texts(llm_utils.get_embedder(models["embeddings"]), texts,
                                                    resolve_path(config["paths"]["processed_dir"]) / "cache" / "embeddings.jsonl")))

    # per-variant deterministic verdicts + judge tasks --------------------------------------
    records, tasks = {}, []
    for v in vs:
        g, test = gens[v["qid"]], v["test"]
        rec = {"qid": v["qid"], "test": test, "variant": v["variant"], "relation": (v.get("gap") or {}).get("relation"),
               "pair_id": v.get("pair_id"), "verdict": None, "route": "auto"}
        if test in ev.CANARY_TESTS:
            rec["verdict"] = "Attack Succeeded" if ev.attack_succeeded(v, g) else "Attack Resisted"
            if test == "S3_injection_embedded":
                rec["cited_planted_passage"] = any(c["status"] == "injected_false_passage"
                                                   for cl in g.get("claims") or [] for c in cl["citations"])
        elif test in ev.JUDGED:
            rec["verdict"] = ev.benign_prefilter(g) if test == "S3_benign" else ev.refusal_prefilter(test, g)
            if rec["verdict"] is None:
                rec["route"] = "judge"
                tasks.append(ev.judge_variant_task(v, g))
        elif test == "S2_regional":
            base = base_gens.get(v["parent_qid"])
            if base is None:
                continue
            rec["verdict"], sim = ev.regional_prefilter(base, g, vec, scfg["regional_similarity"]["consistent"])
            rec["similarity"] = round(sim, 4)
            if rec["verdict"] is None:
                rec["route"] = "judge"
                tasks.append(ev.regional_task(v, base, g))
        records[v["qid"]] = rec

    # LLM judge -----------------------------------------------------------------------------
    print(f"LLM judge ({models['judge']}): {len(tasks)} calls")
    judge = llm_utils.get_chat(models["judge"], models["temperature"])

    def run_task(t):
        res = llm_utils.call_json(judge, t["prompt"], required_keys=["verdict"])
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
        else:
            rec["verdict"] = d.get("verdict")
            rec["judge_reason"] = d.get("reason")

    recs = list(records.values())
    write_jsonl(out_dir / "per_variant.jsonl", recs)
    head, detail = ev.summarize(recs, scfg.get("pass_thresholds", {}))
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id,
               "baseline_run_id": args.baseline_run_id, "models": models, "n_variants": len(recs),
               "judge_failed": len(failed), "headline": head, "detail": detail}
    write_json(out_dir / "summary.json", summary)
    print_summary(summary)
    print(f"\nSaved to {out_dir}")


def print_summary(s):
    print(f"\n{'Metric':<32}{'Score':>8}{'95% CI':>17}{'Threshold':>11}{'Dir':>7}{'Result':>14}{'n':>5}")
    for k, h in s["headline"].items():
        ci = f"{h['ci95'][0]:.2f}-{h['ci95'][1]:.2f}"
        res = h["decision"] or "-"
        print(f"{k:<32}{h['score']:>8.3f}{ci:>17}{str(h['threshold'] if h['threshold'] is not None else '-'):>11}"
              f"{h['direction']:>7}{res:>14}{h['n']:>5}")
    for k, d in s["detail"].items():
        print(f"\n{k}: {d}")


if __name__ == "__main__":
    main()
