"""Reliability metrics (framework R1-R4) computed from a variant run and a clean baseline run."""

import math
import re
from collections import Counter
from itertools import combinations
from statistics import mean, pstdev

from src.alignment.prompts import fill
from src.evaluation.performance import cos, judge_task
from src.reliability import prompts as rp
from src.reliability.rules import HARD_CONSTRAINT_PATTERNS, check_format, check_multi_step, multi_step_parts

ABSTAIN = re.compile(r"\b(neither (handbook|document|policy)|(do|does) not (address|mention|cover|contain|provide|include|"
                     r"specify|say)|not (addressed|mentioned|covered|specified)|no (information|mention|details?)|"
                     r"cannot be determined|not possible to (determine|say)|insufficient information)\b", re.I)
CLARIFY = re.compile(r"\b(could you clarify|can you clarify|do you mean|which (type|kind|aspect)|depends on (what|which)|"
                     r"(is|are|question is) ambiguous|several (types|kinds|interpretations)|if you mean|"
                     r"please specify|could refer to|can refer to)\b", re.I)


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    return round(mean(vals), 4) if vals else None


def normalize_text(t):
    return " ".join((t or "").lower().split())


# ------------------------------------------------------------------ R1

def semantic_consistency(groups, vec, tau):
    """R1 semantic: pairwise cosine across the original + paraphrase answers of each question."""
    pairs = [(a, b) for answers in groups.values() for a, b in combinations(answers, 2)]
    if not pairs:
        return None
    return round(sum(cos(vec[a], vec[b]) >= tau for a, b in pairs) / len(pairs), 4)


def repeatability(groups, vec):
    """R1 repeatability: exact-match rate to the most common output, and std of pairwise similarity."""
    em, spread = [], []
    for answers in groups.values():
        norm = [normalize_text(a) for a in answers]
        em.append(Counter(norm).most_common(1)[0][1] / len(norm))
        sims = [cos(vec[a], vec[b]) for a, b in combinations(answers, 2)]
        spread.append(pstdev(sims) if len(sims) > 1 else 0.0)
    return {"exact_match_rate": mean_or_none(em), "similarity_std": mean_or_none(spread), "n_groups": len(groups)}


# ------------------------------------------------------------------ R2

def accuracy_delta(rows, clean, key):
    """Accuracy (P1 semantic score) of perturbed runs vs the clean runs of the same parent questions."""
    out = {}
    for k in sorted({r[key] for r in rows}):
        sub = [r for r in rows if r[key] == k and r["correctness"] is not None and clean.get(r["parent_qid"]) is not None]
        if sub:
            acc = mean(r["correctness"] for r in sub)
            acc_clean = mean(clean[r["parent_qid"]] for r in sub)
            out[k] = {"accuracy": round(acc, 4), "clean_accuracy": round(acc_clean, 4),
                      "delta": round(acc_clean - acc, 4), "n": len(sub)}
    return out


def long_context_summary(rows):
    """R2 long context: accuracy per (length, position), degradation slope, lost-in-the-middle gap."""
    cells = {}
    for r in rows:
        if r["correctness"] is not None:
            cells.setdefault((r["long_context"]["length"], r["long_context"]["position"]), []).append(r["correctness"])
    heat = {f"{L}|{p}": round(mean(v), 4) for (L, p), v in sorted(cells.items())}
    xs = [math.log10(L) for (L, _), v in cells.items() for _ in v]
    ys = [s for v in cells.values() for s in v]
    slope = None
    if len(set(xs)) > 1:
        mx, my = mean(xs), mean(ys)
        slope = round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs), 4)
    by_pos = {p: mean_or_none(s for (L, pp), v in cells.items() if pp == p for s in v) for p in ("start", "middle", "end")}
    gap = None
    if None not in by_pos.values():
        gap = round((by_pos["start"] + by_pos["end"]) / 2 - by_pos["middle"], 4)
    by_len = {L: mean_or_none(s for (LL, _), v in cells.items() if LL == L for s in v) for L in sorted({c[0] for c in cells})}
    return {"heatmap": heat, "by_position": by_pos, "by_length": by_len,
            "slope_per_10x_length": slope, "lost_in_middle_gap": gap, "accuracy": mean_or_none(ys)}


# ------------------------------------------------------------------ judge tasks

def adversarial_task(v, gen, baseline_answer):
    return judge_task("R2_adversarial", v["qid"], fill(
        rp.ADVERSARIAL_BEHAVIOUR, technique=v["technique"], adv_input=v["adv_input"], ground_truth=v["ground_truth"],
        original=baseline_answer or "(not available)", adversarial=gen["answer"]))


def constraint_task(v, gen):
    return judge_task("R3_constraint", v["qid"], fill(rp.CONSTRAINT_JUDGE, instruction=v["instruction"],
                                                      question=v["question"], answer=gen["answer"]))


def multistep_task(v, gen):
    return judge_task("R3_multi_step", v["qid"], fill(rp.MULTISTEP_JUDGE, answer=gen["answer"]))


def abstention_task(v, gen):
    return judge_task("R4_insufficient_evidence", v["qid"], fill(rp.ABSTENTION_JUDGE, question=v["question"],
                                                                 answer=gen["answer"]))


def ambiguity_task(v, gen):
    return judge_task("R4_ambiguity", v["qid"], fill(rp.AMBIGUITY_JUDGE, interpretations="; ".join(v.get("interpretations", [])),
                                                     question=v["question"], answer=gen["answer"]))


# ------------------------------------------------------------------ deterministic pre-filters

def constraint_prefilter(v, gen):
    """Hard constraints decided by regex (True/False); soft ones return None (judge decides)."""
    pattern = HARD_CONSTRAINT_PATTERNS.get(v["rule"])
    if pattern is None:
        return None
    return not pattern.search(gen["answer"] or "")


def abstained(gen, doc_ids):
    """Deterministic abstention: explicit marker in the answer, or every document listed as not covering it."""
    nf = {n["doc_id"] for n in gen.get("not_found") or [] if n.get("doc_id")}
    return bool(ABSTAIN.search(gen.get("answer") or "")) and (nf >= set(doc_ids) or not gen.get("cited_child_ids"))


def over_abstained(gold, gen):
    """Control set: the model says documents don't cover a point although gold evidence exists for them."""
    gold_docs = {e["doc_id"] for e in gold["evidence"]}
    nf = {n["doc_id"] for n in gen.get("not_found") or [] if n.get("doc_id")}
    return bool(gold_docs) and gold_docs <= nf


def format_ok(v, gen, labels):
    return check_format(v["rule"], gen.get("answer") or "", labels)


def multistep_deterministic(gen, labels):
    checks = check_multi_step(gen.get("answer") or "", labels)
    return checks, multi_step_parts(gen.get("answer") or "") is not None
