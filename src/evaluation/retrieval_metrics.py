"""Retrieval metrics computed purely from chunk IDs (no LLM): Recall@k, Precision@k, Hit@k, MRR, nDCG@k.

Relevance comes from the golden dataset's evidence (child_id / parent_id per document).
Metrics are computed per document side, at child level and parent level. A side whose
document does NOT contain the answer (asymmetric "absent" side) has no relevant chunks, so it
is left out of these averages (absence handling is scored separately in Phase 4).
"combined" = mean of the two document sides for that question.
"""

import math
from statistics import mean


def ranked_lists(retrieved_rows):
    """Ranked child IDs and de-duplicated ranked parent IDs from one document's retrieval."""
    children = [r["child_id"] for r in retrieved_rows]
    parents = []
    for r in retrieved_rows:
        if r["parent_id"] not in parents:
            parents.append(r["parent_id"])
    return children, parents


def at_k(ranked, relevant, k):
    """Recall, precision, hit and nDCG at cut-off k for one ranked list."""
    top = ranked[:k]
    hits = [x for x in top if x in relevant]
    dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(top) if x in relevant)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return {
        "recall": len(hits) / len(relevant),
        "precision": len(hits) / len(top) if top else 0.0,
        "hit": 1.0 if hits else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def reciprocal_rank(ranked, relevant):
    """1 / rank of the first relevant item (0 if none retrieved)."""
    return next((1 / i for i, x in enumerate(ranked, 1) if x in relevant), 0.0)


def question_metrics(gold, retrieval, doc_ids, ks):
    """All retrieval metrics for one question. Keys look like 'D1.child_recall@5' / 'combined.child_mrr'."""
    out = {}
    sides = []
    for d in doc_ids:
        rel_children = {e["child_id"] for e in gold["evidence"] if e["doc_id"] == d}
        rel_parents = {e["parent_id"] for e in gold["evidence"] if e["doc_id"] == d}
        if not rel_children:
            continue                                   # absent side: nothing to retrieve
        sides.append(d)
        children, parents = ranked_lists(retrieval["retrieved"][d])
        for level, ranked, rel in (("child", children, rel_children), ("parent", parents, rel_parents)):
            out[f"{d}.{level}_mrr"] = reciprocal_rank(ranked, rel)
            for k in ks:
                for name, value in at_k(ranked, rel, k).items():
                    out[f"{d}.{level}_{name}@{k}"] = value

    out["_sides_with_evidence"] = sides
    if not sides:
        return out
    for key in {k.split(".", 1)[1] for k in out if not k.startswith("_")}:
        out[f"combined.{key}"] = mean(out[f"{d}.{key}"] for d in sides)
    if len(sides) == len(doc_ids):                     # both documents hold evidence
        for k in ks:
            out[f"both_sides.child_hit@{k}"] = min(out[f"{d}.child_hit@{k}"] for d in doc_ids)
    return out


def aggregate(per_question):
    """Mean of every metric over the questions where it applies, with the count n."""
    keys = sorted({k for q in per_question for k in q if not k.startswith("_")})
    return {k: {"mean": round(mean(q[k] for q in per_question if k in q), 4),
                "n": sum(k in q for q in per_question)} for k in keys}


def summary_table(agg, doc_ids, ks):
    """Small text table of the headline retrieval metrics."""
    lines = [f"{'metric':<22}" + "".join(f"{'@' + str(k):>9}" for k in ks)]
    for scope in doc_ids + ["combined"]:
        for level in ("child", "parent"):
            for m in ("recall", "precision", "ndcg"):
                row = [agg.get(f"{scope}.{level}_{m}@{k}", {}).get("mean") for k in ks]
                lines.append(f"{scope + ' ' + level + ' ' + m:<22}" + "".join(f"{v:>9.3f}" if v is not None else f"{'-':>9}" for v in row))
    both = [agg.get(f"both_sides.child_hit@{k}", {}).get("mean") for k in ks]
    lines.append(f"{'both-sides child hit':<22}" + "".join(f"{v:>9.3f}" if v is not None else f"{'-':>9}" for v in both))
    mrr = agg.get("combined.child_mrr", {}).get("mean")
    lines.append(f"combined child MRR: {mrr}")
    return "\n".join(lines)
