"""Check that the embedding setup works - local model only, no LLM or API calls, nothing in data/ is changed.

API keys are blanked while this runs, so any hidden call to an embedding API would fail the check.
Checks:
  1. config points to a local sentence-transformers model; API embedding names are rejected
  2. the model loads, gives the expected vector size, vectors are normalised
  3. similarity makes sense (related texts score higher than unrelated ones)
  4. the disk cache is reused on a second call
  5. Chroma stores build from the Phase 0 chunks (temporary folder) and stay isolated per document
  6. retrieval quality on the golden questions (Recall@k against golden evidence), if Phase 2 has run
  7. the Phase 1 and Phase 4 code paths that use embeddings run

Usage: python scripts/check_embeddings.py [--expected-dim 384]
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["OPENAI_API_KEY"] = ""          # blank before the .env is loaded: no API can be used
os.environ["GOOGLE_API_KEY"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.alignment import steps as align
from src.common import llm as llm_utils
from src.common.utils import load_config, read_jsonl, resolve_path
from src.evaluation import retrieval_metrics as rm
from src.evaluation.performance import cos
from src.rag import retriever, store

results = []


def check(name, ok, detail=""):
    """Record and print one check result."""
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser(description="Check the local embedding setup (no API calls)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--expected-dim", type=int, default=384)
    args = ap.parse_args()

    config = load_config(args.config)
    models = llm_utils.model_settings(config)
    name = models["embeddings"]
    doc_ids = [config["alignment"]["doc_a"], config["alignment"]["doc_b"]]
    processed = resolve_path(config["paths"]["processed_dir"])
    tmp = Path(tempfile.mkdtemp(prefix="embcheck_"))
    cache = tmp / "embeddings.jsonl"
    print(f"Embedding model: {name}\nTemporary folder: {tmp}\n")

    # 1. config ------------------------------------------------------------------------------
    check("config uses a local sentence-transformers model", llm_utils.is_local_embedding(name), name)
    try:
        llm_utils.get_embedder("text-embedding-3-large")
        check("API embedding models are rejected", False, "text-embedding-3-large was accepted")
    except ValueError:
        check("API embedding models are rejected", True)

    # 2. load + vector size + normalisation ------------------------------------------------------
    t = time.time()
    emb = llm_utils.get_embedder(name)
    v = llm_utils.embed_texts(emb, ["Full-time employees receive 25 days' annual leave."], cache)
    check("model loads and embeds", v.shape[0] == 1, f"{time.time() - t:.1f}s")
    check(f"vector size is {args.expected_dim}", v.shape[1] == args.expected_dim, f"got {v.shape[1]}")
    check("vectors are normalised", abs(float(np.linalg.norm(v[0])) - 1) < 1e-3)

    # 3. similarity sanity --------------------------------------------------------------------------
    a, b, c = llm_utils.embed_texts(emb, ["How many days of annual leave do employees get?",
                                         "What is the yearly holiday entitlement for staff?",
                                         "The office printer is on the third floor."], cache)
    check("related texts score higher than unrelated", cos(a, b) > cos(a, c) + 0.1,
          f"related {cos(a, b):.2f} vs unrelated {cos(a, c):.2f}")

    # 4. cache -------------------------------------------------------------------------------------
    lines = len(read_jsonl(cache))
    llm_utils.embed_texts(emb, ["How many days of annual leave do employees get?"], cache)
    check("cached vectors are reused", len(read_jsonl(cache)) == lines)

    # 5. Chroma stores ------------------------------------------------------------------------------
    children = {d: read_jsonl(processed / d / "children.jsonl") for d in doc_ids}
    if not all(children.values()):
        check("Phase 0 chunks available", False, "run scripts/run_phase0_ingest.py first")
        return finish(tmp)
    embed_fn = lambda texts: llm_utils.embed_texts(emb, texts, cache)
    stores = {}
    for d in doc_ids:
        stores[d], _ = store.build_or_load(d, children[d], embed_fn, name, tmp / "vectorstore",
                                           config["rag"].get("embed_section_title", True))
        check(f"Chroma store {d} built", stores[d]._collection.count() == len(children[d]),
              f"{len(children[d])} chunks")
    qv = llm_utils.embed_texts(emb, ["annual leave entitlement"], cache)[0]
    res = retriever.retrieve(qv, stores, 5)
    check("stores stay isolated per document", retriever.isolation_ok(res))

    # 6. retrieval quality on the golden set -----------------------------------------------------------
    golden = [q for q in read_jsonl(resolve_path(config["golden"]["out_dir"]) / "golden.jsonl") if q.get("evidence")]
    if golden:
        ks = config["rag"].get("ks", [1, 3, 5, 10])
        vecs = llm_utils.embed_texts(emb, [q["question"] for q in golden], cache)
        per_q = [rm.question_metrics(q, {"retrieved": retriever.retrieve(v, stores, max(ks))}, doc_ids, ks)
                 for q, v in zip(golden, vecs)]
        agg = rm.aggregate([{k: x for k, x in p.items() if not k.startswith("_")} for p in per_q])
        print("\n" + rm.summary_table(agg, doc_ids, ks) + "\n")
        gate = config["rag"].get("retrieval_gate", {})
        value = agg.get(f"combined.{gate.get('metric')}", {}).get("mean")
        check(f"retrieval gate ({gate.get('metric')} >= {gate.get('min')})",
              value is not None and value >= gate.get("min", 0), f"{value} on {len(golden)} golden questions")
    else:
        print("(no golden dataset yet - retrieval quality check skipped)")

    # 7. Phase 1 + Phase 4 code paths ------------------------------------------------------------------
    parents = {d: read_jsonl(processed / d / "parents.jsonl") for d in doc_ids}
    va = align.parent_vectors(parents[doc_ids[0]], emb, cache)
    vb = align.parent_vectors(parents[doc_ids[1]], emb, cache)
    cands = align.section_candidates(parents[doc_ids[0]], va, parents[doc_ids[1]], vb,
                                     config["alignment"].get("section_candidates", 3),
                                     config["alignment"].get("section_min_similarity", 0.45))
    with_cand = sum(bool(c) for c in cands.values())
    check("Phase 1 section candidates", with_cand > 0, f"{with_cand}/{len(cands)} sections have candidates")
    x, y = llm_utils.embed_texts(emb, ["Both handbooks give 2 weeks paternity leave.",
                                       "Paternity leave is two weeks in each handbook."], cache)
    check("Phase 4 cosine pre-filter path", 0 < cos(x, y) <= 1.0001, f"similarity {cos(x, y):.2f}")
    finish(tmp)


def finish(tmp):
    print(f"\n{sum(results)}/{len(results)} checks passed. Temporary files: {tmp} (safe to delete)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
