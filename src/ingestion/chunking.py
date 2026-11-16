"""Turn loaded elements into parent (section) chunks and child (retrieval) chunks.

Parents  = top-level sections of the document. The section level is detected, not fixed:
           it is the most important heading rank that occurs at least
           `parent_heading_min_count` times. Smaller headings stay inside the parent
           as sub-headings. Oversized sections are split into "(part N)" parents.
           Documents with no headings at all fall back to fixed-size parents.
Children = small pieces of a parent used for vector search. Text is split with a
           recursive splitter; tables are split by whole rows, repeating the header row.

IDs are deterministic (same document -> same IDs):  D1-P03  and  D1-P03-C02
"""

from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.common.utils import estimate_tokens, short_hash

SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


# ------------------------------------------------------------------ sections

def choose_parent_rank(elements, min_count=2):
    """Pick the heading rank that defines a section: the largest rank seen >= min_count times."""
    counts = Counter(e["rank"] for e in elements if e["type"] == "heading")
    if not counts:
        return None
    for rank in sorted(counts, reverse=True):
        if counts[rank] >= min_count:
            return rank
    return max(counts)


def build_sections(elements, min_count=2):
    """Group elements into sections. Content before the first section heading becomes its own
    section (titled by its first heading, e.g. the document title, or 'Front matter')."""
    parent_rank = choose_parent_rank(elements, min_count)
    if parent_rank is None:
        return [{"title": "Document", "elements": list(elements)}], None

    sections, current = [], {"title": None, "elements": []}
    for e in elements:
        starts_section = e["type"] == "heading" and e["rank"] >= parent_rank
        if starts_section and _has_body(current):
            sections.append(current)
            current = {"title": None, "elements": []}
        # title = first heading of the section; a real section-level heading overrides a
        # bigger one above it with no body (e.g. document title directly followed by "1. Scope")
        if starts_section and (current["title"] is None or e["rank"] == parent_rank):
            current["title"] = e["text"]
        current["elements"].append(e)
    if current["elements"]:
        sections.append(current)

    for s in sections:
        if s["title"] is None:
            first_heading = next((e["text"] for e in s["elements"] if e["type"] == "heading"), None)
            s["title"] = first_heading or "Front matter"
    return sections, parent_rank


def _has_body(section):
    """True if the section already holds something other than headings."""
    return any(e["type"] != "heading" for e in section["elements"])


def split_large_section(section, max_tokens):
    """Split a section bigger than max_tokens into parts, preferring breaks at sub-headings."""
    elements = []
    for e in section["elements"]:                      # first break up single huge elements
        elements.extend(_split_element(e, max_tokens))

    parts, current, size = [], [], 0
    for e in elements:
        n = estimate_tokens(e["text"])
        at_subheading = e["type"] == "heading" and size >= max_tokens * 0.5
        if current and (size + n > max_tokens or at_subheading):
            parts.append(current)
            current, size = [], 0
        current.append(e)
        size += n
    if current:
        parts.append(current)

    if len(parts) == 1:
        return [{"title": section["title"], "elements": parts[0]}]
    return [{"title": f"{section['title']} (part {i})", "elements": p} for i, p in enumerate(parts, 1)]


def _split_element(e, max_tokens):
    """Break one oversized text/table element into smaller elements of the same type."""
    if estimate_tokens(e["text"]) <= max_tokens:
        return [e]
    if e["type"] == "table":
        return [dict(e, text=t, rows=r) for t, r in _table_groups(e["rows"], max_tokens)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=0,
                                              length_function=estimate_tokens, separators=SEPARATORS,
                                              keep_separator="end")
    return [dict(e, text=t) for t in splitter.split_text(e["text"])]


# ------------------------------------------------------------------ parents

def make_parents(elements, doc, cfg):
    """Build parent chunks for one document. `doc` is its config entry, `cfg` the ingestion config."""
    sections, parent_rank = build_sections(elements, cfg.get("parent_heading_min_count", 2))
    max_tokens = cfg.get("parent_max_tokens", 1500)

    if parent_rank is None:                            # no headings found anywhere
        sections = split_large_section(sections[0], max_tokens)
        sections = [dict(s, title=f"Part {i}") for i, s in enumerate(sections, 1)]
    else:
        sections = [part for s in sections for part in split_large_section(s, max_tokens)]

    parents = []
    for i, s in enumerate(sections, 1):
        text = "\n\n".join(e["text"] for e in s["elements"])
        parents.append({
            "parent_id": f"{doc['id']}-P{i:02d}",
            "doc_id": doc["id"],
            "doc_label": doc.get("label", doc["id"]),
            "order": i,
            "title": s["title"],
            "pages": _pages(s["elements"]),
            "n_tokens": estimate_tokens(text),
            "text": text,
            "text_hash": short_hash(text),
            "elements": s["elements"],                 # kept for child building; dropped when saved
        })
    return parents, parent_rank


# ------------------------------------------------------------------ children

def make_children(parent, cfg):
    """Split one parent into child chunks. Sub-headings start a new child; tables are split by rows."""
    size = cfg.get("child_tokens", 120)
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=cfg.get("child_overlap_tokens", 20),
                                              length_function=estimate_tokens, separators=SEPARATORS,
                                              keep_separator="end")
    pieces = []          # (text, kind, source elements)
    buffer = []

    def flush():
        nonlocal buffer
        if buffer:
            text = "\n\n".join(e["text"] for e in buffer)
            for piece in splitter.split_text(text):
                pieces.append((piece, "text", _source_elements(piece, buffer)))
        buffer = []

    for e in parent["elements"]:
        if e["type"] == "heading" and buffer and any(b["type"] != "heading" for b in buffer):
            flush()
            buffer = [e]
        elif e["type"] == "table":
            caption = " / ".join(b["text"] for b in buffer) if buffer and all(b["type"] == "heading" for b in buffer) else ""
            if caption:
                buffer = []                            # heading(s) right above a table become its caption
            flush()
            for text, _ in _table_groups(e["rows"], size, caption):
                pieces.append((text, "table", [e]))
        else:
            buffer.append(e)
    flush()
    pieces = _merge_small_pieces(pieces, cfg.get("child_min_tokens", 25))

    children = []
    for i, (text, kind, src) in enumerate(pieces, 1):
        children.append({
            "child_id": f"{parent['parent_id']}-C{i:02d}",
            "parent_id": parent["parent_id"],
            "doc_id": parent["doc_id"],
            "doc_label": parent["doc_label"],
            "section_title": parent["title"],
            "order": i,
            "kind": kind,
            "pages": _pages(src),
            "n_tokens": estimate_tokens(text),
            "text": text,
            "text_hash": short_hash(text),
        })
    return children


def _merge_small_pieces(pieces, min_tokens):
    """Glue tiny text pieces (e.g. a lone heading) onto the next text piece, or the previous one if last."""
    merged = []
    carry = None
    for text, kind, src in pieces:
        if carry and kind == "text":
            text, src = carry[0] + "\n\n" + text, carry[2] + src
            carry = None
        elif carry:                                    # next piece is a table: keep the small piece as is
            merged.append(carry)
            carry = None
        if kind == "text" and estimate_tokens(text) < min_tokens:
            carry = (text, kind, src)
        else:
            merged.append((text, kind, src))
    if carry:
        if merged and merged[-1][1] == "text":
            prev = merged.pop()
            merged.append((prev[0] + "\n\n" + carry[0], "text", prev[2] + carry[2]))
        else:
            merged.append(carry)
    return merged


def _table_groups(rows, max_tokens, caption=""):
    """Yield (markdown, rows) groups of whole table rows; every group repeats the header row."""
    from src.ingestion.loaders import table_to_markdown

    header, body = rows[0], rows[1:]
    prefix = caption + "\n" if caption else ""
    if not body:
        yield prefix + table_to_markdown([header]), [header]
        return
    group = []
    for row in body:
        candidate = [header] + group + [row]
        if group and estimate_tokens(prefix + table_to_markdown(candidate)) > max_tokens:
            yield prefix + table_to_markdown([header] + group), [header] + group
            group = []
        group.append(row)
    yield prefix + table_to_markdown([header] + group), [header] + group


def _source_elements(piece, elements):
    """Elements whose text overlaps a child piece (used to work out the child's page numbers)."""
    hits = [e for e in elements if e["text"][:60] in piece or piece[:60] in e["text"] or e["text"][-60:] in piece]
    return hits or elements


def _pages(elements):
    """Sorted unique page numbers of the given elements (empty for formats without pages)."""
    return sorted({e["page"] for e in elements if e.get("page") is not None})
