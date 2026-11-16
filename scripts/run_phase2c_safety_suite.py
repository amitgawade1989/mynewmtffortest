"""Phase 2c - LLM-generated safety test suite (framework S2 Regional, S3 Refusal Appropriateness).

Questions come from regional policy gaps: Phase 1 fact pairs where one handbook differs from or lacks
what the other has. Same flow as the golden set (Phase 2), every LLM step resumable in data/safety/work/:
  1. generate   one generator call per gap writes a realistic, hard case per category from the gap's facts
  2. check      deterministic: length, {{CANARY}} placeholder present, no document metadata
  3. critic     LLM judge: right category? realism 1-5, difficulty 1-5
  4. revise     rejected cases are rewritten with the feedback, then re-checked (max_revisions)
Categories per gap:
  S3 Benign / Ambiguous      legitimate gap question, grey-area "is there a proper route", vague wording (PARTIAL)
  S3 Genuinely Unsafe        subtle employee misuse, overt forgery / deception, employer exploiting the gap
  S3 Injection Direct        claimed authority in the question + canary token
  S3 Injection Embedded      planted "policy update" passage + canary, with an innocent question
  S3 Instruction Hierarchy   user message attacking a fixed system rule + canary
  S3 Role-play / Hypothetical  optional framework rows (config: safety.optional)
S2 Regional re-asks golden questions (LLM-written in Phase 2) with the two handbooks in reverse order.
Region names are allowed here (unlike the golden and reliability suites).

Outputs in data/safety/:
  variants.jsonl        accepted cases (+ S2 Regional) for Phase 3
  rejected.jsonl        cases that never passed, with every round's issues
  suite_review.md       examples per test + rejection reasons
  suite_manifest.json   counts, models, rejection reasons
  with --limit-gaps the same files get a _sample suffix and the full suite is untouched
Then:   python scripts/run_phase3_rag.py --questions data/safety/variants.jsonl --run-id <id>

Usage:
  python scripts/run_phase2c_safety_suite.py                  # run / resume
  python scripts/run_phase2c_safety_suite.py --limit-gaps 2   # quick look at a few gaps
  python scripts/run_phase2c_safety_suite.py --retry-failed   # re-run gaps whose LLM calls failed
  python scripts/run_phase2c_safety_suite.py --fresh          # delete previous Phase 2c output first
"""

import argparse
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import llm as llm_utils
from src.common.runner import run_resumable
from src.common.utils import load_config, read_json, read_jsonl, resolve_path, write_json, write_jsonl
from src.reliability.build import stratified_sample
from src.safety import build


def main():
    ap = argparse.ArgumentParser(description="Phase 2c: LLM-generated safety test suite")
    ap.add_argument("--config", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--limit-gaps", type=int, default=None, help="only the first N gaps (quick test)")
    args = ap.parse_args()

    config = load_config(args.config)
    scfg = config["safety"]
    gcfg = scfg.get("generation") or {}
    seed = scfg.get("seed", 11)
    out = resolve_path(scfg["out_dir"])
    work = out / "work"
    doc_a, doc_b = config["alignment"]["doc_a"], config["alignment"]["doc_b"]
    labels = {d["id"]: d.get("label", d["id"]) for d in config["documents"]}
    regions = {d["id"]: d.get("region") or d.get("label", d["id"]) for d in config["documents"]}
    processed = resolve_path(config["paths"]["processed_dir"])
    golden = read_jsonl(resolve_path(config["golden"]["out_dir"]) / "golden.jsonl")
    pairs = read_jsonl(processed / "alignment" / "fact_pairs.jsonl")
    if not pairs or not golden:
        sys.exit("Fact pairs or golden dataset missing - run Phases 1 and 2 first.")

    if args.fresh and out.exists():
        shutil.rmtree(out)
    work.mkdir(parents=True, exist_ok=True)

    models = llm_utils.model_settings(config)
    gen_model, critic_model = gcfg.get("model") or models["generator"], gcfg.get("critic_model") or models["judge"]
    fp_path, fingerprint = work / "models.json", {"generator": gen_model, "critic": critic_model}
    if fp_path.exists() and read_json(fp_path) != fingerprint:       # never mix models in one suite
        sys.exit(f"Models changed since the last Phase 2c run ({read_json(fp_path)}). Re-run with --fresh.")
    write_json(fp_path, fingerprint)

    gaps = build.select_gaps(pairs, scfg, config["golden"].get("exclude_pair_flags", []), doc_a, doc_b,
                             labels, regions, seed)
    print(f"Regional gaps available: {len(gaps)} ({dict(Counter(g['relation'] for g in gaps))})")
    jobs = []
    for i, g in enumerate(gaps[:args.limit_gaps] if args.limit_gaps else gaps):
        cats = build.categories_for(i, g, scfg)
        if cats:
            jobs.append({"gap": g, "cats": cats})

    generator = llm_utils.get_chat(gen_model, gcfg.get("temperature", 0.7))
    critic = llm_utils.get_chat(critic_model, models["temperature"])
    print(f"\nGenerating: {len(jobs)} gaps | generator {gen_model} | critic {critic_model}")
    settings = f"{gen_model}|{critic_model}|{gcfg.get('min_realism', 4)}|{gcfg.get('min_difficulty', 3)}|{gcfg.get('max_revisions', 1)}"
    key = lambda j: f"{j['gap']['pair_id']}|{','.join(j['cats'])}|{settings}"

    def run(j):
        values = build.gap_values(j["gap"], labels, doc_a, doc_b, regions)
        return build.generate_cases(j["gap"], j["cats"], generator, critic, values, gcfg)

    try:
        rows, failed = run_resumable(jobs, key, run, work / "llm_cases.jsonl", work / "failed_llm_cases.jsonl",
                                     args.retry_failed, "gaps", workers=gcfg.get("workers", 4))
    except llm_utils.QuotaExhausted:
        sys.exit("Stopped: quota exhausted. Re-run later to resume.")

    by_key = {r["key"]: r for r in rows}
    prefix = scfg.get("canary_prefix", "CNRY")
    variants, rejected = [], []
    for j in jobs:
        r = by_key.get(key(j))
        if r is None:
            continue
        variants += build.llm_variants(j["gap"], r, prefix)
        rejected += [{"pair_id": j["gap"]["pair_id"], "category": c, "issue": why,
                      "history": [h for h in r.get("history", []) if h["category"] == c]}
                     for c, why in r["rejected"].items()]
    asked = sum(len(j["cats"]) for j in jobs)
    print(f"   kept {len(variants)} cases of {asked} requested | rejected {len(rejected)} | gaps failed {len(failed)}"
          + (" (re-run with --retry-failed)" if failed else ""))
    if not variants:
        sys.exit("No safety cases were generated (see the errors above) - existing suite files left unchanged.")
    variants += build.regional_variants(stratified_sample(golden, scfg["regional"]["n_questions"], seed, "regional"))

    sfx = "_sample" if args.limit_gaps else ""              # a quick test never replaces the full suite
    write_jsonl(out / f"variants{sfx}.jsonl", variants)
    write_jsonl(out / f"rejected{sfx}.jsonl", rejected)
    counts = dict(Counter(v["test"] for v in variants))
    write_json(out / f"suite_manifest{sfx}.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": seed, "models": fingerprint,
        "generation": gcfg, "n_gaps": len(jobs), "cases_requested": asked, "n_variants": len(variants),
        "by_test": counts, "gaps_failed": len(failed),
        "rejected_by_category": dict(Counter(r["category"] for r in rejected))})
    write_review(out / f"suite_review{sfx}.md", variants, rejected)
    print(f"\n{len(variants)} variants: {counts}\nSaved to {out / f'variants{sfx}.jsonl'}")


def write_review(path, variants, rejected):
    """A few examples per test and variant for a human check, plus why the critic rejected cases."""
    lines = ["# Safety suite - examples", ""]
    for test in sorted({v["test"] for v in variants}):
        vs = [v for v in variants if v["test"] == test]
        lines += [f"## {test} ({len(vs)} variants)", ""]
        for tag in sorted({v["variant"] for v in vs}):
            for v in [v for v in vs if v["variant"] == tag][:2]:
                score = f" _(realism {v['realism']}, difficulty {v['difficulty']})_" if v.get("realism") else ""
                lines.append(f"- `{v['qid']}`{score}: {v['question']}")
                if v.get("injected_passage"):
                    lines.append(f"  _planted in {v['injected_passage']['doc_id']}:_ {v['injected_passage']['text']}")
                if v.get("system_rule"):
                    lines.append(f"  _system rule:_ {v['system_rule']}")
        lines.append("")
    if rejected:
        lines += [f"## Rejected by checks / critic ({len(rejected)})", ""]
        lines += [f"- `{r['pair_id']}` {r['category']}: {r['issue']}" for r in rejected]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
