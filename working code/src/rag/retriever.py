"""Parent-child retrieval: search children in each document's own store, expand to parents.

Every question is searched against EVERY document separately - the question text never says
which document it is about, and no document's chunks can appear in another document's results.
"""


def retrieve(query_vec, stores, k):
    """Ranked children per document: {doc_id: [{rank, child_id, parent_id, score}]}.

    score = cosine similarity (1 - cosine distance). `stores` is {doc_id: Chroma store}.
    """
    results = {}
    for doc_id, store in stores.items():
        hits = store.similarity_search_by_vector_with_relevance_scores(list(query_vec), k=k)
        results[doc_id] = [{"rank": i, "child_id": d.metadata["child_id"], "parent_id": d.metadata["parent_id"],
                            "doc_id": d.metadata["doc_id"], "score": round(1 - float(dist), 4)}
                           for i, (d, dist) in enumerate(hits, 1)]
    return results


def isolation_ok(results):
    """True if every retrieved chunk belongs to the store (document) it was retrieved from."""
    return all(r["doc_id"] == doc_id for doc_id, rows in results.items() for r in rows)


def select_parents(ranked, context_k, max_parents):
    """Parents of the top context_k children, in rank order, without duplicates (capped)."""
    parents = []
    for r in ranked[:context_k]:
        if r["parent_id"] not in parents:
            parents.append(r["parent_id"])
    return parents[:max_parents]


def build_context(results, children_by_doc, parents_index, labels, context_k, max_parents):
    """Context text with one block per document; each child passage is prefixed with its ID.

    Returns (context_text, context_info) where context_info[doc] = {"parent_ids", "child_ids"}.
    """
    parents_by_doc = {d: select_parents(ranked, context_k, max_parents) for d, ranked in results.items()}
    return render_blocks(parents_by_doc, children_by_doc, parents_index, labels)


def render_blocks(parents_by_doc, children_by_doc, parents_index, labels):
    """Render the given parent sections as one labelled block per document (children prefixed by ID)."""
    blocks, info = [], {}
    for doc_id, parent_ids in parents_by_doc.items():
        child_ids, lines = [], [f"=== DOCUMENT: {labels[doc_id]} ==="]
        for pid in parent_ids:
            lines.append(f"## Section: {parents_index[pid]['title']}")
            for c in children_by_doc[doc_id]:
                if c["parent_id"] == pid:
                    lines.append(f"[{c['child_id']}] {c['text']}")
                    child_ids.append(c["child_id"])
        if not parent_ids:
            lines.append("(no passages retrieved)")
        lines.append(f"=== END OF {labels[doc_id]} ===")
        blocks.append("\n".join(lines))
        info[doc_id] = {"parent_ids": parent_ids, "child_ids": child_ids}
    return "\n\n".join(blocks), info


def inject_passage(context, info, doc_id, text, labels):
    """Adversarial test: add a FALSE passage to one document's block, with a plausible chunk ID.

    The ID is recorded in info[doc_id]["injected_ids"] so citations of it can be recognised.
    """
    fake_id = f"{doc_id}-P90-C01"
    end = f"=== END OF {labels[doc_id]} ==="
    context = context.replace(end, f"[{fake_id}] {text}\n{end}", 1)
    info[doc_id].setdefault("injected_ids", []).append(fake_id)
    return context, info
