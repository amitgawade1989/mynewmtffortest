"""Small shared helpers: config, paths, JSON/JSONL I/O, token estimate, hashing."""

import hashlib
import json
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    """Load config.yaml (project root by default) and return it as a dict."""
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p):
    """Return an absolute Path; relative paths are resolved from the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def estimate_tokens(text):
    """Rough token count (~4 characters per token). Used only for chunk sizing."""
    return (len(text) + 3) // 4 if text else 0


def short_hash(text, n=10):
    """Short SHA-1 of a string, used to detect when a chunk's text has changed."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def file_sha256(path):
    """SHA-256 of a file, used to detect when a source document has changed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    """Write obj as pretty JSON, creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path):
    """Read a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, rows):
    """Overwrite a JSONL file with the given rows (one dict per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    """Append one row and flush it to disk immediately, so a crash loses nothing already done."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    """Read a JSONL file into a list of dicts. A half-written last line (from a crash) is skipped."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] skipping unreadable line in {path.name}")
    return rows
