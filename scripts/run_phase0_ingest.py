"""Phase 0 - ingest documents into parent/child chunks.

For every document in config.yaml:
  1. load it into elements (headings / text / tables)
  2. build parent chunks (sections) and child chunks
  3. run validation checks
  4. save to data/processed/<doc_id>/ :
       elements.jsonl        raw extracted elements (for debugging)
       parents.jsonl         parent chunks
       children.jsonl        child chunks
       structure_review.md   human-readable outline to review before Phase 1
  5. write data/processed/ingest_manifest.json (file hashes, counts, checks, errors)

Usage:  python scripts/run_phase0_ingest.py [--config config.yaml]
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.utils import file_sha256, load_config, resolve_path, write_json, write_jsonl
from src.ingestion.chunking import make_children, make_parents
from src.ingestion.loaders import load_document


def ingest_document(doc, cfg, out_dir):
    """Run the full Phase 0 flow for one document and return its manifest entry."""
    path = resolve_path(doc["path"])
    elements, info = load_document(path, cfg)
    parents, parent_rank = make_parents(elements, doc, cfg)
    children = [c for p in parents for c in make_children(p, cfg)]
    checks = validate(elements, parents, children, cfg)

    doc_dir = out_dir / doc["id"]
    write_jsonl(doc_dir / "elements.jsonl", elements)
    write_jsonl(doc_dir / "parents.jsonl", [{k: v for k, v in p.items() if k != "elements"} for p in parents])
    write_jsonl(doc_dir / "children.jsonl", children)
    write_review(doc_dir / "structure_review.md", doc, info, parent_rank, parents, children, checks)

    return {
        "id": doc["id"], "label": doc.get("label"), "path": str(doc["path"]),
        "sha256": file_sha256(path), "status": "ok", "info": info,
        "section_heading_rank": parent_rank,
        "n_parents": len(parents), "n_children": len(children), "checks": checks,
    }


def validate(elements, parents, children, cfg):
    """Sanity checks on the chunks. Returns a dict of results; 'passed' is False if anything is wrong."""
    words = lambda s: set(re.findall(r"\w+", s.lower()))
    source_words = set().union(*(words(e["text"]) for e in elements)) if elements else set()
    child_words = set().union(*(words(c["text"]) for c in children)) if children else set()
    coverage = len(source_words & child_words) / len(source_words) if source_words else 0.0

    ids = [p["parent_id"] for p in parents] + [c["child_id"] for c in children]
    parents_without_children = [p["parent_id"] for p in parents if not any(c["parent_id"] == p["parent_id"] for c in children)]
    limit = cfg.get("child_tokens", 120) * 1.5
    oversized = [c["child_id"] for c in children if c["n_tokens"] > limit]

    checks = {
        "unique_ids": len(ids) == len(set(ids)),
        "parents_without_children": parents_without_children,
        "word_coverage": round(coverage, 4),
        "oversized_children": oversized,          # a single long table row can cause this; warning only
    }
    checks["passed"] = checks["unique_ids"] and not parents_without_children and coverage >= 0.99
    return checks


def write_review(path, doc, info, parent_rank, parents, children, checks):
    """Write a short markdown outline of the detected structure for a human to check."""
    lines = [f"# Structure review - {doc.get('label', doc['id'])} ({doc['id']})", "",
             f"- File: `{doc['path']}`",
             f"- Heading detection: {info.get('heading_mode')} | body font size: {info.get('body_font_size', '-')} "
             f"| section heading rank: {parent_rank}",
             f"- Elements: {info['n_elements']} | headings: {info['n_headings']} | tables: {info['n_tables']}",
             f"- Removed headers/footers: {info.get('removed_header_footer', [])}",
             f"- Parents: {len(parents)} | children: {len(children)} | checks passed: {checks['passed']} "
             f"(word coverage {checks['word_coverage']:.1%})", "",
             "| Parent | Title | Pages | Tokens | Children |", "|---|---|---|---|---|"]
    for p in parents:
        n = sum(c["parent_id"] == p["parent_id"] for c in children)
        lines.append(f"| {p['parent_id']} | {p['title']} | {p['pages']} | {p['n_tokens']} | {n} |")
    lines.append("")
    for p in parents:
        lines.append(f"## {p['parent_id']} - {p['title']}")
        for c in children:
            if c["parent_id"] == p["parent_id"]:
                preview = c["text"].replace("\n", " ")[:110]
                lines.append(f"- `{c['child_id']}` [{c['kind']}, {c['n_tokens']} tok] {preview}...")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Phase 0: ingest documents into parent/child chunks")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["ingestion"]
    out_dir = resolve_path(config["paths"]["processed_dir"])

    doc_ids = [d["id"] for d in config["documents"]]
    if len(doc_ids) != len(set(doc_ids)) or not all(re.fullmatch(r"[A-Za-z0-9_]+", i) for i in doc_ids):
        sys.exit("Document ids in config.yaml must be unique and contain only letters, digits or _")

    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "settings": cfg, "documents": []}
    for doc in config["documents"]:
        print(f"\n== {doc['id']} ({doc.get('label', '')}): {doc['path']}")
        try:
            entry = ingest_document(doc, cfg, out_dir)
            c = entry["checks"]
            print(f"   {entry['info']['heading_mode']} headings | {entry['n_parents']} parents | "
                  f"{entry['n_children']} children | {entry['info']['n_tables']} tables | "
                  f"coverage {c['word_coverage']:.1%} | checks {'PASSED' if c['passed'] else 'FAILED'}")
            if c["oversized_children"]:
                print(f"   warning: oversized children {c['oversized_children']}")
        except Exception as exc:                       # one bad document must not stop the others
            print(f"   ERROR: {exc}")
            entry = {"id": doc["id"], "path": str(doc["path"]), "status": "error", "error": repr(exc)}
        manifest["documents"].append(entry)

    write_json(out_dir / "ingest_manifest.json", manifest)
    print(f"\nSaved to {out_dir}")
    if any(d["status"] != "ok" or not d["checks"]["passed"] for d in manifest["documents"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
