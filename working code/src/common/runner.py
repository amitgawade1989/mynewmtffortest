"""Resumable processing of a list of items with crash-safe output.

Each finished item is appended to `out_path` immediately. On a re-run, items whose key is
already in `out_path` are skipped, so an interrupted run continues where it stopped.
Items that still fail after retries go to `failed_path` (with the error) and the run moves on.
A QuotaExhausted error stops the run straight away - just re-run later to resume.
"""

from datetime import datetime, timezone

from src.common.llm import QuotaExhausted
from src.common.utils import append_jsonl, read_jsonl, write_jsonl


def run_resumable(items, key_fn, process_fn, out_path, failed_path, retry_failed=False, label="items", workers=1):
    """Process items (in parallel if workers > 1). process_fn(item) returns a dict saved with the item's key.

    retry_failed=True re-processes only the items listed in failed_path.
    Returns (done_rows, failed_rows) read back from disk - only rows whose key belongs to the
    current items, so results saved for items that no longer exist (stale keys) are ignored.
    """
    current = {key_fn(it) for it in items}
    done = {r["key"] for r in read_jsonl(out_path)} & current
    failed_rows = read_jsonl(failed_path)
    failed_keys = {r["key"] for r in failed_rows} & current

    if retry_failed:
        todo = [it for it in items if key_fn(it) in failed_keys and key_fn(it) not in done]
        write_jsonl(failed_path, [r for r in failed_rows if r["key"] not in {key_fn(it) for it in todo}])
    else:
        todo = [it for it in items if key_fn(it) not in done and key_fn(it) not in failed_keys]

    skipped_failed = len([it for it in items if key_fn(it) in failed_keys]) if not retry_failed else 0
    print(f"   {label}: {len(items)} total | {len(done)} already done | {len(todo)} to run"
          + (f" | {skipped_failed} previously failed (use --retry-failed)" if skipped_failed else ""))

    def save(n, key, result=None, exc=None):
        """Write one outcome (always from the main thread, so files are never written concurrently)."""
        if exc is None:
            append_jsonl(out_path, {"key": key, "ts": _now(), **result})
            print(f"   [{n}/{len(todo)}] {key} ok")
        else:
            append_jsonl(failed_path, {"key": key, "ts": _now(), "error": f"{type(exc).__name__}: {str(exc)[:3000]}"})
            print(f"   [{n}/{len(todo)}] {key} FAILED: {str(exc)[:150]}")

    if workers <= 1:
        for n, item in enumerate(todo, 1):
            try:
                save(n, key_fn(item), process_fn(item))
            except QuotaExhausted as exc:
                print(f"   quota exhausted - stopping. Re-run later to resume. ({str(exc)[:120]})")
                raise
            except Exception as exc:
                save(n, key_fn(item), exc=exc)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_fn, item): key_fn(item) for item in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                try:
                    save(n, futures[fut], fut.result())
                except QuotaExhausted as exc:
                    print(f"   quota exhausted - stopping. Re-run later to resume. ({str(exc)[:120]})")
                    for f in futures:
                        f.cancel()
                    raise
                except Exception as exc:
                    save(n, futures[fut], exc=exc)

    keep = lambda rows: [r for r in rows if r["key"] in current]
    return keep(read_jsonl(out_path)), keep(read_jsonl(failed_path))


def _now():
    """Current UTC time as ISO text."""
    return datetime.now(timezone.utc).isoformat()
