"""Lightweight shim for ingest_sources.gather_ingest

This file provides a minimal, defensive implementation so the monolithic
server can import `gather_ingest`. It reads `data/insights.ndjson` if
present and returns a list of parsed JSON objects. The real project may
replace this with a richer ingestion pipeline.
"""
from typing import List, Dict, Any, Optional
import os, json

def gather_ingest(max_total: Optional[int] = None) -> List[Dict[str, Any]]:
    """Collect items for ingestion.

    Intended to be callable as a synchronous helper (the server sometimes
    calls it via `asyncio.to_thread(gather_ingest, ...)`).

    Args:
        max_total: optional maximum number of items to return.

    Returns:
        A list of dict items (possibly empty).
    """
    path = os.path.join(os.path.dirname(__file__), 'data', 'insights.ndjson')
    # Fallback to top-level data/ if package layout places it in workspace root
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'insights.ndjson')
    items = []
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        items.append(obj)
                        if max_total is not None and len(items) >= int(max_total):
                            break
                    except Exception:
                        # ignore malformed lines
                        continue
    except Exception:
        # If reading fails for any reason, return empty list silently.
        return []
    return items


if __name__ == '__main__':
    # simple standalone smoke-run
    items = gather_ingest()
    print(f"Gathered {len(items)} items")
    if items:
        print("First item keys:", list(items[0].keys()))