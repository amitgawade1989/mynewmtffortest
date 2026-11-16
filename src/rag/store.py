"""Vector store: one Chroma collection per document, so retrieval can never mix documents.

Children are embedded and stored (with child_id / parent_id / doc_id metadata).
Parents are not embedded; they are looked up by ID from Phase 0's parents.jsonl.
A small index manifest records which chunks + embedding model each collection was built from;
if either changes, the collection is rebuilt automatically.
"""

from src.common.utils import read_json, read_jsonl, short_hash, write_json


def numpy2_compat():
    """Let older chromadb (0.5.x) import under numpy 2.

    chromadb 0.5.x still references np.float_ (removed in numpy 2.0); importing it then fails with
    "`np.float_` was removed in the NumPy 2.0 release". Restoring the alias as np.float64 (what it
    always meant) fixes the import. No effect on numpy 1.x or on chromadb versions that don't need it.
    """
    import numpy as np
    if not hasattr(np, "float_"):
        np.float_ = np.float64


def child_embed_text(child, with_title=True):
    """Text that is embedded for a child: optionally prefixed with its section title."""
    return f"{child['section_title']}\n{child['text']}" if with_title else child["text"]


def open_store(doc_id, store_dir):
    """Open (or create) the Chroma collection for one document, using cosine distance.

    No embedding function is attached: vectors are computed by src.common.llm.embed_texts
    (local MiniLM or an API model, with a disk cache) and passed in / queried by vector.
    """
    numpy2_compat()                                 # must run before chromadb is imported
    from langchain_chroma import Chroma

    return Chroma(collection_name=f"doc_{doc_id}", embedding_function=None,
                  persist_directory=str(store_dir / doc_id), collection_metadata={"hnsw:space": "cosine"})


def build_or_load(doc_id, children, embed_fn, embed_model, store_dir, with_title=True, force=False):
    """Return the store for doc_id, (re)building it if the chunks or embedding model changed.

    embed_fn(list_of_texts) -> list of vectors.
    """
    fingerprint = {"embed_model": embed_model, "with_title": with_title,
                   "chunks": short_hash("".join(c["child_id"] + c["text_hash"] for c in children), 16)}
    manifest_path = store_dir / doc_id / "index_manifest.json"
    store = open_store(doc_id, store_dir)
    up_to_date = (manifest_path.exists() and read_json(manifest_path) == fingerprint
                  and store._collection.count() == len(children))
    if up_to_date and not force:
        return store, False

    store.delete_collection()                       # also needed when the vector size changes (e.g. 3072 -> 384)
    store = open_store(doc_id, store_dir)
    texts = [child_embed_text(c, with_title) for c in children]
    store._collection.add(
        ids=[c["child_id"] for c in children],
        embeddings=[[float(x) for x in v] for v in embed_fn(texts)],   # plain Python floats, not numpy
        documents=texts,
        metadatas=[{"child_id": c["child_id"], "parent_id": c["parent_id"], "doc_id": c["doc_id"],
                    "section_title": c["section_title"]} for c in children],
    )
    write_json(manifest_path, fingerprint)
    return store, True


def load_corpus(processed_dir, doc_ids):
    """Phase 0 chunks for the given documents: (children_by_doc, children_index, parents_index)."""
    children_by_doc = {d: read_jsonl(processed_dir / d / "children.jsonl") for d in doc_ids}
    children_index = {c["child_id"]: c for cs in children_by_doc.values() for c in cs}
    parents_index = {p["parent_id"]: p for d in doc_ids for p in read_jsonl(processed_dir / d / "parents.jsonl")}
    return children_by_doc, children_index, parents_index
