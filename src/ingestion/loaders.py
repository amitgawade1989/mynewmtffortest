"""Load a document (PDF / DOCX / TXT / MD) into a flat list of "elements".

Every loader returns the same shape, so later steps never care about file type:
    {"type": "heading", "text": str, "rank": float, "page": int|None}
    {"type": "text",    "text": str, "page": int|None}
    {"type": "table",   "text": markdown str, "rows": [[cell, ...], ...], "page": int|None}

`rank` orders headings by importance (bigger = more important). For PDFs it is the
font size; for DOCX/MD/TXT it is derived from the heading level or numbering depth.
No document-specific headings or section names are hard-coded anywhere.
"""

import re
import unicodedata
from collections import Counter
from pathlib import Path

BULLETS = {"•", "▪", "◦", "●", "‣", "∙", "-", "–", "\x7f"}

# Generic heading patterns used only when font/style information is missing:
# "1.", "2.3", "2.3.1", "Section 4", "Chapter IV", "Article 2", "IV.", "Appendix A"
NUMBERED_HEADING = re.compile(
    r"^(?:(?P<num>\d+(?:\.\d+)*)\.?|(?:section|chapter|article|part|appendix|schedule)\s+[\dA-Z]+[.:]?|[IVXLC]+\.)\s+\S",
    re.IGNORECASE,
)


def load_document(path, cfg):
    """Load any supported file and return (elements, info). `cfg` is the `ingestion` config block."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        elements, info = load_pdf(path, cfg)
    elif ext == ".docx":
        elements, info = load_docx(path, cfg)
    elif ext in (".txt", ".md"):
        elements, info = load_text(path, cfg)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    elements = [e for e in elements if e["text"].strip()]
    info["n_elements"] = len(elements)
    info["n_headings"] = sum(e["type"] == "heading" for e in elements)
    info["n_tables"] = sum(e["type"] == "table" for e in elements)
    return elements, info


# ------------------------------------------------------------------ text helpers

def clean_text(text):
    """Normalise unicode (ligatures, odd spaces), unify bullets and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x7f", "•").replace("­", "")
    text = re.sub(r"[ \t ]+", " ", text)
    return text.strip()


def join_lines(lines):
    """Join wrapped lines into one paragraph, repairing words hyphenated across line breaks."""
    out = ""
    for line in lines:
        if not out:
            out = line
        elif out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out = out + " " + line
    return out


def heading_rank_from_text(text):
    """Return a rank for text that *looks* like a heading (numbering / ALL CAPS), else None.

    Used as a fallback when font-size or style information is not available.
    "1. Intro" -> 99, "1.2 Sub" -> 98, "Section 3" -> 99, short ALL CAPS line -> 99.5.
    """
    words = text.split()
    if not words or len(words) > 15 or text.endswith((".", ",", ";", ":")):
        return None
    m = NUMBERED_HEADING.match(text)
    if m:
        depth = m.group("num").count(".") + 1 if m.group("num") else 1
        return 100 - depth
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 4 and all(c.isupper() for c in letters):
        return 99.5
    return None


def table_to_markdown(rows):
    """Render a list of rows as a markdown table (first row is the header)."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [[c.replace("|", "/").strip() for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


# ------------------------------------------------------------------ PDF

def load_pdf(path, cfg):
    """Read a PDF with PyMuPDF: drop repeated headers/footers, keep tables, detect headings by font size."""
    import pymupdf

    doc = pymupdf.open(path)
    pages = []
    for page in doc:
        page_dict = page.get_text("dict", sort=True)
        tables = _pdf_tables(page, page_dict) if cfg.get("extract_tables", True) else []
        lines = []
        for block_no, block in enumerate(page_dict["blocks"]):
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans or _inside_any(line["bbox"], tables):
                    continue
                lines.append({
                    "text": clean_text("".join(s["text"] for s in line["spans"])),
                    "size": round(max(s["size"] for s in spans), 1),
                    "bold": all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans),
                    "y0": line["bbox"][1], "y1": line["bbox"][3],
                    "block": block_no, "page": page.number + 1,
                    "page_height": page.rect.height,
                })
        pages.append({"lines": lines, "tables": tables})

    repeated = _repeated_lines(pages, cfg.get("header_footer_page_ratio", 0.5),
                               cfg.get("header_footer_margin", 0.12))
    for p in pages:
        p["lines"] = [l for l in p["lines"] if _line_key(l["text"]) not in repeated]

    body_size = _body_font_size(pages)
    delta = cfg.get("heading_min_size_delta", 1.0)
    max_words = cfg.get("heading_max_words", 20)

    def by_font(line):
        if line["size"] >= body_size + delta and len(line["text"].split()) <= max_words:
            return line["size"]
        return None

    def by_pattern(line):
        return heading_rank_from_text(line["text"]) if line["bold"] else None

    elements = _pdf_elements(pages, by_font)
    mode = "font-size"
    if not any(e["type"] == "heading" for e in elements):
        elements = _pdf_elements(pages, by_pattern)   # no larger fonts: try bold numbered lines
        mode = "bold-numbering"
    info = {"file_type": "pdf", "pages": len(doc), "body_font_size": body_size,
            "heading_mode": mode, "removed_header_footer": sorted(repeated)}
    return elements, info


def _pdf_tables(page, page_dict):
    """Find tables and rebuild each cell from the full text spans.

    find_tables() gives the grid, but it clips text that overflows a cell, so cell
    text is rebuilt from the spans that *start* inside each row/column.
    """
    try:
        found = page.find_tables().tables
    except Exception:
        return []
    spans = [s for b in page_dict["blocks"] for l in b.get("lines", []) for s in l["spans"] if s["text"].strip()]
    tables = []
    for t in found:
        col_x = sorted({round(c[0]) for r in t.rows for c in r.cells if c})
        if len(col_x) < 2 or len(t.rows) < 2:
            continue
        rows = []
        for r in t.rows:
            ry0, ry1 = r.bbox[1], r.bbox[3]
            cells = [[] for _ in col_x]
            for s in spans:
                x0, y0, x1, y1 = s["bbox"]
                yc = (y0 + y1) / 2
                if not (ry0 - 1 <= yc <= ry1 + 1 and t.bbox[0] - 2 <= x0 <= t.bbox[2]):
                    continue
                col = max(i for i, x in enumerate(col_x) if x <= x0 + 2) if x0 + 2 >= col_x[0] else 0
                cells[col].append((y0, x0, s["text"]))
            rows.append([clean_text(" ".join(txt for _, _, txt in sorted(c))) for c in cells])
        rows = [r for r in rows if any(r)]
        tables.append({"bbox": tuple(t.bbox), "y0": t.bbox[1], "rows": rows, "page": page.number + 1})
    return tables


def _inside_any(bbox, tables):
    """True if the centre of bbox lies inside any table's bounding box."""
    xc, yc = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(t["bbox"][0] - 1 <= xc <= t["bbox"][2] + 1 and t["bbox"][1] - 1 <= yc <= t["bbox"][3] + 1
               for t in tables)


def _line_key(text):
    """Normalise a line for header/footer matching ("Page 3" and "Page 4" become the same key)."""
    return re.sub(r"\d+", "#", text.lower()).strip()


def _repeated_lines(pages, min_ratio, margin):
    """Keys of lines that sit in the top/bottom margin and repeat on many pages (headers/footers)."""
    if len(pages) < 2:
        return set()
    seen = Counter()
    for p in pages:
        keys = {_line_key(l["text"]) for l in p["lines"]
                if l["y1"] <= l["page_height"] * margin or l["y0"] >= l["page_height"] * (1 - margin)}
        seen.update(keys)
    return {k for k, n in seen.items() if n / len(pages) >= min_ratio and n >= 2}


def _body_font_size(pages):
    """Most common font size weighted by amount of text = the body text size."""
    sizes = Counter()
    for p in pages:
        for l in p["lines"]:
            sizes[l["size"]] += len(l["text"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _pdf_elements(pages, heading_rank_fn):
    """Turn page lines + tables into ordered elements using the given heading rule."""
    elements = []
    for p in pages:
        items = [(l["y0"], "line", l) for l in p["lines"]] + [(t["y0"], "table", t) for t in p["tables"]]
        items.sort(key=lambda x: x[0])          # stable: keeps reading order of lines
        para, para_block, para_page = [], None, None

        def flush():
            nonlocal para
            if para:
                elements.append({"type": "text", "text": join_lines(para), "page": para_page})
            para = []

        prev_was_heading = False
        for _, kind, item in items:
            if kind == "table":
                flush()
                elements.append({"type": "table", "text": table_to_markdown(item["rows"]),
                                 "rows": item["rows"], "page": item["page"]})
                prev_was_heading = False
                continue
            rank = heading_rank_fn(item)
            if rank is not None:
                flush()
                last = elements[-1] if elements else None
                # a heading wrapped over two lines -> one heading
                if prev_was_heading and last["rank"] == rank and last["page"] == item["page"]:
                    last["text"] += " " + item["text"]
                else:
                    elements.append({"type": "heading", "text": item["text"], "rank": rank, "page": item["page"]})
                prev_was_heading = True
                continue
            prev_was_heading = False
            only_bullet = len(para) == 1 and para[0] in BULLETS
            if para and item["block"] != para_block and not only_bullet:
                flush()
            if not para:
                para_page = item["page"]
            para.append(item["text"])
            para_block = item["block"]
        flush()
    return elements


# ------------------------------------------------------------------ DOCX

def load_docx(path, cfg):
    """Read a DOCX in body order. Headings come from 'Heading N'/'Title' styles, else numbering patterns."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(path)
    elements = []
    for child in d.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "tbl":
            table = Table(child, d)
            rows = [[clean_text(c.text) for c in r.cells] for r in table.rows]
            rows = [r for r in rows if any(r)]
            if rows:
                elements.append({"type": "table", "text": table_to_markdown(rows), "rows": rows, "page": None})
        elif tag == "p":
            para = Paragraph(child, d)
            text = clean_text(para.text)
            if not text:
                continue
            style = (para.style.name or "").lower() if para.style is not None else ""
            rank = None
            if style == "title":
                rank = 101
            elif style.startswith("heading"):
                level = re.findall(r"\d+", style)
                rank = 100 - int(level[0]) if level else 99
            elif para.runs and all(r.bold for r in para.runs if r.text.strip()):
                rank = heading_rank_from_text(text)
            if rank is not None:
                elements.append({"type": "heading", "text": text, "rank": rank, "page": None})
            else:
                elements.append({"type": "text", "text": text, "page": None})
    return elements, {"file_type": "docx", "heading_mode": "styles/numbering"}


# ------------------------------------------------------------------ TXT / MD

def load_text(path, cfg):
    """Read plain text or markdown. Headings from '#' markers, else numbering / ALL CAPS lines."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    elements, para, table_rows = [], [], []

    def flush():
        nonlocal para, table_rows
        if para:
            elements.append({"type": "text", "text": join_lines(para), "page": None})
        if table_rows:
            elements.append({"type": "table", "text": table_to_markdown(table_rows), "rows": table_rows, "page": None})
        para, table_rows = [], []

    for line in raw.splitlines():
        line = clean_text(line)
        if not line:
            flush()
            continue
        if line.startswith("|"):                                   # markdown table row
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                if para:
                    elements.append({"type": "text", "text": join_lines(para), "page": None})
                    para = []
                table_rows.append(cells)
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        rank = 100 - len(m.group(1)) if m else heading_rank_from_text(line)
        if rank is not None:
            flush()
            elements.append({"type": "heading", "text": m.group(2) if m else line, "rank": rank, "page": None})
        else:
            if table_rows:
                flush()
            para.append(line)
    flush()
    return elements, {"file_type": Path(path).suffix.lstrip("."), "heading_mode": "markdown/numbering"}
