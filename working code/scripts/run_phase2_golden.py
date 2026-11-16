"""Phase 2 - golden dataset generation.

Flow (every LLM step is resumable; results are saved item by item in data/golden/work/):
  1. generate   questions per topic + a cross-topic batch, from verified fact pairs
  2. check      deterministic: metadata lint, type vs relations, both docs covered, quotes, ...
  3. critic     LLM judge scores answerable / correct / unambiguous / needs both docs / no leaks
  4. revise     rejected questions are rewritten with the feedback, then re-checked (max_revisions)
Outputs in data/golden/:
  golden.jsonl         accepted questions with ground truth, key points and traceable evidence
  rejected.jsonl       questions that never passed, with every round's issues
  golden_review.md     readable table for spot-checking
  golden_manifest.json counts, rejection reasons, token usage

Usage:
  python scripts/run_phase2_golden.py                 # run / resume
  python scripts/run_phase2_golden.py --retry-failed  # re-run only LLM calls that failed
  python scripts/run_phase2_golden.py --fresh         # delete previous Phase 2 output first
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
from src.golden import steps
from src.golden.lint import build_banned


def main():
    ap = argparse.ArgumentParser(description="Phase 2: golden dataset")
    ap.add_argument("--config", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    gcfg, acfg = config["golden"], config["alignment"]
    models = llm_utils.model_settings(config)
    processed = resolve_path(config["paths"]["processed_dir"])
    out = resolve_path(gcfg["out_dir"])
    work = out / "work"
    doc_a, doc_b = acfg["doc_a"], acfg["doc_b"]
    doc_ids = [doc_a, doc_b]
    labels = {d["id"]: d.get("label", d["id"]) for d in config["documents"]}

    if args.fresh and out.exists():
        shutil.rmtree(out)
    work.mkdir(parents=True, exist_ok=True)

    # never mix results from different models in one dataset
    fp_path, fingerprint = work / "models.json", {"generator": models["generator"], "judge": models["judge"]}
    if fp_path.exists() and read_json(fp_path) != fingerprint:
        sys.exit(f"Models changed since the last Phase 2 run ({read_json(fp_path)}). Re-run with --fresh.")
    write_json(fp_path, fingerprint)

    # inputs from Phase 0 + 1
    pairs_all = read_jsonl(processed / "alignment" / "fact_pairs.jsonl")
    if not pairs_all:
        sys.exit("Phase 1 output missing - run scripts/run_phase1_align.py first.")
    pairs = steps.usable_pairs(pairs_all, gcfg.get("exclude_pair_flags", []))
    pairs_index = {p["pair_id"]: p for p in pairs}
    children_index = {c["child_id"]: c for d in doc_ids for c in read_jsonl(processed / d / "children.jsonl")}
    parents_index = {p["parent_id"]: p for d in doc_ids for p in read_jsonl(processed / d / "parents.jsonl")}
    topics = read_json(processed / "alignment" / "topic_map.json")["topics"]
    banned = build_banned(config)
    print(f"Usable fact pairs: {len(pairs)} of {len(pairs_all)} | relations {dict(Counter(p['relation'] for p in pairs))}")

    generator = llm_utils.get_chat(models["generator"], models["temperature"])
    judge = llm_utils.get_chat(models["judge"], models["temperature"])
    cross_model = gcfg.get("cross_topic_model") or models["generator"]
    cross_generator = llm_utils.get_chat(cross_model, models["temperature"])
    gen_for = lambda kind: cross_generator if kind == "cross" else generator
    rf = args.retry_failed

    try:
        # 1. generate ------------------------------------------------------------
        print("\n[1/3] Generating questions")
        items = []
        for t in topics:
            tp = [p for p in pairs if p["topic_id"] == t["topic_id"]]
            if tp:
                items.append({"item_id": t["topic_id"], "kind": "topic", "topic": t["name"], "pairs": tp,
                              "topic_ids": [t["topic_id"]], "n": gcfg.get("questions_per_topic", 8)})
        for n, (ta, tb) in enumerate(steps.cross_topic_combos([i["topic_ids"][0] for i in items],
                                                              gcfg.get("cross_topic_questions", 0)), 1):
            items.append({"item_id": f"XT{n}", "kind": "cross", "topic": f"{ta} + {tb}", "topic_ids": [ta, tb],
                          "pairs": [p for p in pairs if p["topic_id"] in (ta, tb)], "n": 2, "model": cross_model})
        gen_rows, _ = run_resumable(items, steps.generation_key,
                                    lambda it: steps.generate_for_topic(it, gen_for(it["kind"]), gcfg, labels, doc_a, doc_b),
                                    work / "generated.jsonl", work / "failed_generate.jsonl", rf, "topics")

        cands = {}
        for row in gen_rows:
            item = next(i for i in items if i["item_id"] == row["item_id"])
            for n, q in enumerate(row["questions"], 1):
                qid = f"Q-{row['item_id']}-{n:02d}"
                c = steps.make_candidate(q, qid, item["kind"], pairs_index, children_index, doc_ids, banned)
                c["item_topic_ids"] = item["topic_ids"]
                cands[qid] = c
        print(f"   {len(cands)} candidate questions")

        # 2-3. check + critic, with revision rounds ------------------------------
        history = {qid: [] for qid in cands}
        accepted, rejected, pending = {}, {}, dict(cands)
        max_rev = gcfg.get("max_revisions", 2)
        for rnd in range(max_rev + 1):
            print(f"\n[2/3] Review round {rnd} - {len(pending)} questions")
            to_critic = [c for c in pending.values() if not c["check_issues"]]
            crit_rows, _ = run_resumable(to_critic, steps.candidate_key,
                                         lambda c: steps.critique(c, judge, children_index, parents_index, labels, doc_a, doc_b),
                                         work / "critic.jsonl", work / "failed_critic.jsonl", rf, "critic")
            crit_by_key = {r["key"]: r["critic"] for r in crit_rows}

            failed_now = {}
            for qid, c in pending.items():
                key = steps.candidate_key(c)
                if c["check_issues"]:
                    issues = c["check_issues"]
                elif key not in crit_by_key:
                    issues = ["critic call failed - re-run with --retry-failed"]
                else:
                    issues = steps.critic_issues(crit_by_key[key], gcfg.get("critic_min_score", 4))
                history[qid].append({"revision": c["revision"], "question": c["question"], "issues": issues})
                if issues:
                    failed_now[qid] = (c, issues)
                else:
                    accepted[qid] = steps.golden_record(c, crit_by_key[key])
            print(f"   accepted so far: {len(accepted)} | rejected this round: {len(failed_now)}")

            if rnd == max_rev or not failed_now:
                rejected = failed_now
                break

            # 4. revise ------------------------------------------------------------
            print(f"\n[3/3] Revising {len(failed_now)} rejected questions")
            to_revise = [dict(c, _issues=iss) for c, iss in failed_now.values()]

            def revise_one(c):
                rp = [p for p in pairs if p["topic_id"] in c["item_topic_ids"]]
                return steps.revise(c, c["_issues"], gen_for(c["source"]), rp, labels, doc_a, doc_b)

            rev_rows, _ = run_resumable(to_revise, steps.candidate_key, revise_one,
                                        work / "revisions.jsonl", work / "failed_revisions.jsonl", rf, "revisions")
            rev_by_key = {r["key"]: r for r in rev_rows}
            pending = {}
            for c, iss in failed_now.values():
                r = rev_by_key.get(steps.candidate_key(c))
                if not r or not isinstance(r["question"], dict):
                    rejected[c["qid"]] = (c, iss + ["revision call failed"])
                    continue
                nc = steps.make_candidate(r["question"], c["qid"], c["source"], pairs_index, children_index,
                                          doc_ids, banned, revision=c["revision"] + 1)
                nc["item_topic_ids"] = c["item_topic_ids"]
                pending[c["qid"]] = nc
    except llm_utils.QuotaExhausted:
        sys.exit("Stopped: quota exhausted. Re-run the same command later to resume.")

    # outputs ----------------------------------------------------------------------
    golden = [accepted[q] for q in sorted(accepted)]
    rej = [{**{k: c[k] for k in steps.FIELDS + ["qid", "source", "revision", "evidence", "absent"]},
            "final_issues": iss, "history": history[q]} for q, (c, iss) in sorted(rejected.items())]
    write_jsonl(out / "golden.jsonl", golden)
    write_jsonl(out / "rejected.jsonl", rej)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "models": models,
        "n_candidates": len(cands), "n_accepted": len(golden), "n_rejected": len(rej),
        "by_type": dict(Counter(g["question_type"] for g in golden)),
        "by_reasoning": dict(Counter(g["reasoning"] for g in golden)),
        "by_difficulty": dict(Counter(g["difficulty"] for g in golden)),
        "by_topic": dict(Counter(t for g in golden for t in g["topic_ids"])),
        "accepted_after_revision": sum(g["revision"] > 0 for g in golden),
        "rejection_reasons": dict(Counter(reason_label(i) for r in rej for i in r["final_issues"]
                                          if not i.startswith("critic note")).most_common(15)),
        "token_usage": token_usage(work),
    }
    write_json(out / "golden_manifest.json", manifest)
    write_review(out / "golden_review.md", golden, rej, manifest)
    print(f"\nAccepted {len(golden)} | rejected {len(rej)} | by type {manifest['by_type']}")
    print(f"Saved to {out}")


def reason_label(issue):
    """Short category for an issue, e.g. 'critic: correct' or 'question text reveals metadata'."""
    if issue.startswith("critic:"):
        return "critic: " + issue.split()[1]
    return issue.split(":")[0][:60]


def token_usage(work):
    """Total tokens over all saved LLM responses of this phase."""
    total = Counter()
    for f in work.glob("*.jsonl"):
        for r in read_jsonl(f):
            u = r.get("usage") or {}
            total["input_tokens"] += u.get("input_tokens", 0)
            total["output_tokens"] += u.get("output_tokens", 0)
            total["calls"] += 1 if u else 0
    return dict(total)


def write_review(path, golden, rejected, manifest):
    """Readable markdown for spot-checking the dataset."""
    cell = lambda s: str(s or "").replace("|", "/").replace("\n", " ")
    lines = ["# Golden dataset review", "",
             f"- Accepted: {manifest['n_accepted']} of {manifest['n_candidates']} "
             f"({manifest['accepted_after_revision']} after revision) | rejected: {manifest['n_rejected']}",
             f"- By type: {manifest['by_type']}", f"- By reasoning: {manifest['by_reasoning']}",
             f"- By difficulty: {manifest['by_difficulty']}", f"- Rejection reasons: {manifest['rejection_reasons']}",
             "", "## Accepted", "",
             "| QID | Type | Reasoning | Diff. | Question | Short answer | Evidence (child ids) | Absent |",
             "|---|---|---|---|---|---|---|---|"]
    for g in golden:
        ev = ", ".join(sorted({e["child_id"] for e in g["evidence"]}))
        ab = ", ".join(f"{a['doc_id']}: {a['attribute']}" for a in g["absent"])
        lines.append(f"| {g['qid']} | {g['question_type']} | {g['reasoning']} | {g['difficulty']} | {cell(g['question'])} "
                     f"| {cell(g['short_answer'])} | {ev} | {cell(ab)} |")
    lines += ["", "## Rejected", "", "| QID | Question | Final issues |", "|---|---|---|"]
    for r in rejected:
        lines.append(f"| {r['qid']} | {cell(r['question'])} | {cell('; '.join(r['final_issues']))[:400]} |")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
