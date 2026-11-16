"""Build reliability variants (R1-R4) from the golden dataset.

Every variant is a question record for the RAG runner: {qid, parent_qid, test, variant, question, ...}.
Variants of golden questions copy the parent's gold fields so they can be scored the same way.
"""

from src.alignment.prompts import fill
from src.common.llm import call_json
from src.golden.lint import lint_question
from src.reliability import prompts as rp
from src.reliability.perturb import PERTURBATIONS, rng_for
from src.reliability.rules import CONSTRAINT_RULES, FORMAT_RULES, MULTI_STEP

GOLD_FIELDS = ["question_type", "reasoning", "difficulty", "topics", "ground_truth", "short_answer",
               "key_points", "pair_ids", "relations", "expected_values", "evidence", "absent"]


def stratified_sample(golden, n, seed, key):
    """n golden questions, spread evenly over question types, deterministic for (seed, key)."""
    rng = rng_for(seed, key)
    groups = {}
    for q in golden:
        groups.setdefault(q["question_type"], []).append(q)
    for g in groups.values():
        rng.shuffle(g)
    out, i = [], 0
    while len(out) < min(n, len(golden)):
        for g in groups.values():
            if i < len(g) and len(out) < n:
                out.append(g[i])
        i += 1
    return out


def variant(parent, test, tag, question, **extra):
    """A variant record that inherits the parent's gold fields."""
    rec = {"qid": f"{parent['qid']}~{tag}", "parent_qid": parent["qid"], "test": test, "variant": tag,
           "question": question, **{k: parent.get(k) for k in GOLD_FIELDS}}
    rec.update(extra)
    return rec


# ------------------------------------------------------------------ deterministic variants

def typo_variants(parents, cfg, seed):
    """R2 typo / noise: character typos, casing noise, whitespace noise."""
    rates = {"typo_char": cfg["char_rate"], "typo_case": cfg["case_rate"], "typo_space": cfg["space_rate"]}
    return [variant(p, "R2_typo_noise", kind, fn(p["question"], rates[kind], rng_for(seed, p["qid"], kind)))
            for p in parents for kind, fn in PERTURBATIONS.items()]


def repeat_variants(parents, n_runs):
    """R1 repeatability: the identical question n_runs times."""
    return [variant(p, "R1_repeatability", f"rep{i}", p["question"]) for p in parents for i in range(1, n_runs + 1)]


def format_variants(parents):
    """R3 format: each rule in each of its phrasings appended to the question."""
    return [variant(p, "R3_format", f"fmt_{rule}_{i}", f"{p['question']} {text}", rule=rule)
            for p in parents for rule, phrasings in FORMAT_RULES.items() for i, text in enumerate(phrasings, 1)]


def constraint_variants(parents):
    """R3 constraint: each content restriction appended to the question."""
    return [variant(p, "R3_constraint", f"con_{rule}", f"{p['question']} {text}", rule=rule, instruction=text)
            for p in parents for rule, text in CONSTRAINT_RULES.items()]


def multi_step_variants(parents):
    """R3 multi-step: the compound instruction in two phrasings."""
    return [variant(p, "R3_multi_step", f"multi_{i}", f"{p['question']} {text}", instruction=text)
            for p in parents for i, text in enumerate(MULTI_STEP, 1)]


def long_context_variants(parents, cfg):
    """R2 long context: every length x position (retrieval is bypassed for these)."""
    return [variant(p, "R2_long_context", f"lc_{L}_{pos}", p["question"], long_context={"length": L, "position": pos})
            for p in parents for L in cfg["lengths"] for pos in cfg["positions"]]


# ------------------------------------------------------------------ LLM-generated variants

def paraphrase_variants(parent, llm, n, banned):
    """R1 semantic: n paraphrases (lint-checked; similarity is checked by the caller)."""
    res = call_json(llm, fill(rp.PARAPHRASE, n=n, question=parent["question"]), required_keys=["paraphrases"])
    out = []
    for i, text in enumerate(res["data"]["paraphrases"][:n], 1):
        if isinstance(text, str) and text.strip() and not lint_question(text, banned):
            out.append(variant(parent, "R1_semantic", f"para{i}", text.strip()))
    return {"variants": out, "raw": res["raw"], "usage": res["usage"]}


def adversarial_variants(parent, llm, labels, doc_a, doc_b, banned):
    """R2 adversarial: false premise, leading question, injected false passage."""
    facts = "\n".join(f"- {ev['attribute']}: {labels[doc_a]} = {'; '.join(ev[doc_a]) or 'not addressed'} | "
                      f"{labels[doc_b]} = {'; '.join(ev[doc_b]) or 'not addressed'}" for ev in parent["expected_values"])
    res = call_json(llm, fill(rp.ADVERSARIAL, question=parent["question"], ground_truth=parent["ground_truth"],
                              facts=facts, label_a=labels[doc_a], label_b=labels[doc_b]),
                    required_keys=["false_premise_question", "leading_question", "injected_passage"])
    d, out = res["data"], []
    for tag, key in (("adv_false_premise", "false_premise_question"), ("adv_leading", "leading_question")):
        text = (d.get(key) or "").strip()
        if text and not lint_question(text, banned):
            out.append(variant(parent, "R2_adversarial", tag, text, technique=tag, adv_input=text))
    ip = d.get("injected_passage") or {}
    if isinstance(ip, dict) and (ip.get("text") or "").strip():
        target = doc_a if str(ip.get("target", "A")).upper().startswith("A") else doc_b
        out.append(variant(parent, "R2_adversarial", "adv_injected", parent["question"], technique="adv_injected",
                           adv_input=ip["text"].strip(), injected_passage={"doc_id": target, "text": ip["text"].strip()}))
    return {"variants": out, "raw": res["raw"], "usage": res["usage"]}
