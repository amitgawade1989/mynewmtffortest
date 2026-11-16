"""Build scorecard rows: one per framework sub-metric, with our score, threshold and a key finding.

Rows come from the framework workbook ('Dimension Deep-Dives'), and applicability from the
'Capability Coverage' sheet column for the chosen capability (e.g. Document Comparison).
"""

from openpyxl import load_workbook

# (metric number, sub-metric name in the framework) -> where our score lives
SCORE_MAP = {
    ("P1", "Exact / Objective"): ("perf", "P1_exact"),
    ("P1", "Semantic Correctness"): ("perf", "P1_semantic"),
    ("P2", "Query Relevance"): ("perf", "P2_relevance"),
    ("P3", "Requirement / Relevant Information Coverage"): ("perf", "P3_completeness"),
    ("P5", "Logical Coherence"): ("perf", "P5_logical_coherence"),
    ("P5", "Linguistic Quality"): ("perf", "P5_linguistic_quality"),
    ("P6", "Claim Faithfulness"): ("perf", "P6_claim_faithfulness"),
    ("P6", "Citation Accuracy"): ("perf", "P6_citation_accuracy"),
    ("P7", "Latency"): ("perf", "P7_latency_sla"),
    ("P7", "Token Usage / Cost"): ("perf", "P7_tokens"),
    ("R1", "Semantic"): ("rel", "R1_semantic_consistency"),
    ("R1", "Deterministic Repeatability"): ("rel", "R1_repeatability"),
    ("R2", "Typo / Noise"): ("rel", "R2_typo_noise"),
    ("R2", "Adversarial"): ("rel", "R2_adversarial"),
    ("R2", "Long Context"): ("rel", "R2_long_context"),
    ("R3", "Format"): ("rel", "R3_format"),
    ("R3", "Constraint"): ("rel", "R3_constraint"),
    ("R3", "Multi-step Instruction"): ("rel", "R3_multi_step"),
    ("R4", "Insufficient Evidence"): ("rel", "R4_insufficient_evidence"),
    ("R4", "Ambiguity Handling"): ("rel", "R4_ambiguity"),
    ("S5", "Fabricated Citations"): ("perf", "S5_fabricated_citations"),
}
LOWER_IS_BETTER = {"S5_fabricated_citations"}


def read_framework(path, capability):
    """Framework rows [{dimension, num, metric, sub_metric, description, implementation, measurement, applies}]."""
    wb = load_workbook(path, data_only=True)
    deep = wb["Dimension Deep-Dives"]
    rows, header_row = [], None
    for r in deep.iter_rows(values_only=True):
        if header_row is None:
            if r and r[0] == "Dimension":
                header_row = r
            continue
        if r[0] and r[1]:
            rows.append({"dimension": r[0], "num": r[1], "metric": r[2], "sub_metric": r[3], "description": r[4],
                         "implementation": r[5], "measurement": r[6]})

    cov = wb["Capability Coverage"]
    cov_header, applies = None, {}
    for r in cov.iter_rows(values_only=True):
        if cov_header is None:
            if r and r[0] == "Dimension":
                cov_header = list(r)
                col = cov_header.index(capability)
            continue
        if r[0] and r[1]:
            applies[(r[1], r[3])] = (r[col] or "").strip()
    for row in rows:
        row["applies"] = applies.get((row["num"], row["sub_metric"]), "")
    return rows


def scorecard_rows(framework, perf, rel, optional_cfg, perf_thresholds):
    """Combine framework rows with our results. Status explains rows without a score."""
    out = []
    for f in framework:
        key = SCORE_MAP.get((f["num"], f["sub_metric"]))
        row = dict(f, score=None, threshold=None, direction="higher", n=None, status="", finding="")
        if not f["applies"]:
            row["status"] = "N/A for this capability (framework)"
        elif key is None:
            row["status"] = ("Out of scope (Performance + Reliability run)" if f["dimension"] == "Safety"
                             else "Not run (optional)")
        else:
            source, k = key
            if k == "S5_fabricated_citations":
                rate = perf["document_comparison"].get("S5_fabricated_citation_rate")
                row.update(score=rate, threshold=perf_thresholds.get("S5_fabricated_citations"), direction="lower",
                           status="Evaluated (computed from P6 citation checks)", n=perf["coverage"]["evaluated"])
            elif k == "P7_tokens":
                row.update(status="Evaluated (no threshold)", n=perf["coverage"]["evaluated"])
            else:
                block = (perf["headline"] if source == "perf" else rel["headline"]).get(k)
                if block is None:
                    row["status"] = "Not run (optional)" if f["applies"] == "O" else "Not run"
                else:
                    row.update(score=block["score"], threshold=block.get("threshold"), n=block.get("n"),
                               status="Evaluated")
            row["finding"] = key_finding(k, perf, rel)
        out.append(row)
    return out


def pct(x, d=0):
    return "-" if x is None else f"{x * 100:.{d}f}%"


def key_finding(k, perf, rel):
    """One-line, data-driven explanation for a sub-metric (no hand-written numbers)."""
    sm, dc = perf["sub_metrics"], perf["document_comparison"]
    rd = rel.get("detail", {})
    if k == "P1_exact":
        return (f"Relation exact match {pct(sm['P1_relation_exact_match'], 1)}, numeric match {pct(sm['P1_numeric_match'], 1)}, "
                f"token F1 {sm['P1_token_f1']:.2f}.")
    if k == "P1_semantic":
        return (f"Routes {sm['P1_semantic_routes']}; polarity flips {pct(sm['P1_polarity_flip_rate'], 1)}, "
                f"attribution swaps {pct(dc['attribution_swap_rate_judged'], 1)} of judged answers.")
    if k == "P2_relevance":
        return f"Routes {sm['P2_relevance_routes']}."
    if k == "P3_completeness":
        return "Share of reference key points present and correct (judge-verified after embedding match)."
    if k == "P5_logical_coherence":
        return f"{sm['P5_nli_contradiction_flags']} NLI contradiction flags (cross-document pairs excluded); judge is authoritative."
    if k == "P6_claim_faithfulness":
        return ("Sentence-level NLI vs top-k context chunks; long multi-fact claims are a known limit of the "
                "deterministic method. Absence claims are scored under R4/absence detection.")
    if k == "P6_citation_accuracy":
        return (f"NLI on cited passages. ID match vs golden evidence: child precision {pct(sm['P6_citation_child_precision'], 1)}, "
                f"recall {pct(sm['P6_citation_child_recall'], 1)}; both documents cited {pct(dc['both_documents_cited_rate'], 1)}.")
    if k == "P7_latency_sla":
        t, tot = sm["P7_ttft_s"], sm["P7_total_latency_s"]
        return f"TTFT p50 {t.get('p50')} s / p95 {t.get('p95')} s; total p50 {tot.get('p50')} s / p95 {tot.get('p95')} s."
    if k == "P7_tokens":
        cost = sm.get("P7_total_cost_usd")
        return (f"{sm['P7_tokens_per_answer']:.0f} tokens per answer; "
                + (f"${cost:.4f} for {perf['coverage']['evaluated']} answers." if cost is not None else "cost not priced."))
    if k == "S5_fabricated_citations":
        return f"Chunk IDs that do not exist; citations outside the retrieved context: {pct(dc['citations_outside_context_rate'], 1)}."
    if k == "R1_semantic_consistency" and "R1_semantic" in rd:
        d = rd["R1_semantic"]
        return (f"Expected-match rate vs ground truth (headline); pairwise answer consistency {pct(d['consistency_rate'], 1)}; "
                f"relation label agreement {pct(d['relation_agreement'], 1)}.")
    if k == "R1_repeatability" and "R1_repeatability" in rd:
        d = rd["R1_repeatability"]
        return f"Exact-match rate to modal answer at temperature 0 + fixed seed; similarity std {d['similarity_std']}."
    if k == "R2_typo_noise" and "R2_typo_noise" in rd:
        d = rd["R2_typo_noise"]
        drops = ", ".join(f"{p.replace('typo_', '')} -{v['delta'] * 100:.1f}pp" for p, v in d["by_perturbation"].items())
        return f"Accuracy retained vs clean (1 - drop). Drops: {drops}; fragile: {d['fragile_perturbations'] or 'none'}."
    if k == "R2_adversarial" and "R2_adversarial" in rd:
        d = rd["R2_adversarial"]
        held = ", ".join(f"{t.replace('adv_', '')} {pct(v)}" for t, v in d["held_rate_by_technique"].items())
        return f"Held rate by technique: {held}; planted passage cited in {pct(d['cited_planted_passage_rate'])} of injected cases."
    if k == "R2_long_context" and "R2_long_context" in rd:
        d = rd["R2_long_context"]
        return (f"Accuracy across 1K-100K tokens x start/middle/end; slope {d['slope_per_10x_length']} per 10x length; "
                f"lost-in-middle gap {d['lost_in_middle_gap']}.")
    if k == "R3_format" and "R3_format" in rd:
        return "By rule: " + ", ".join(f"{r} {pct(v)}" for r, v in rd["R3_format"]["compliance_by_rule"].items()) + "."
    if k == "R3_constraint" and "R3_constraint" in rd:
        return "By rule: " + ", ".join(f"{r} {pct(v)}" for r, v in rd["R3_constraint"]["compliance_by_rule"].items()) + "."
    if k == "R3_multi_step" and "R3_multi_step" in rd:
        return "Requirement coverage over 5 sub-requirements (4 deterministic, 1 judged)."
    if k == "R4_ambiguity" and "R4_ambiguity" in rd:
        return f"Verdicts: {rd['R4_ambiguity']['verdicts']}."
    if k == "R4_insufficient_evidence" and "R4_insufficient_evidence" in rd:
        d = rd["R4_insufficient_evidence"]
        return f"Correct abstentions on unanswerable set; false-abstention rate on golden set {pct(d['false_abstention_rate_on_golden'], 1)}."
    return ""
