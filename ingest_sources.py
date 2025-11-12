"""Enhanced ingest_sources.gather_ingest

This file provides ingestion from data/insights.ndjson (primary) and can also
fall back to reading directly from research ingest/ folder if needed.
The modular server imports `gather_ingest` for local research content.
"""
from typing import List, Dict, Any, Optional
import os, json, glob

def gather_ingest(max_total: Optional[int] = None) -> List[Dict[str, Any]]:
    """Collect items for ingestion.

    Intended to be callable as a synchronous helper (the server sometimes
    calls it via `asyncio.to_thread(gather_ingest, ...)`).

    Primary source: data/insights.ndjson (pre-processed research content)
    Fallback: direct reading from research ingest/ folder

    Args:
        max_total: optional maximum number of items to return.

    Returns:
        A list of dict items (possibly empty).
    """
    # Try primary source: data/insights.ndjson
    items = _read_insights_ndjson(max_total)
    
    # If no items and no insights file, try fallback: direct folder reading
    if not items:
        items = _read_research_folder_fallback(max_total)
    
    return items


def _read_insights_ndjson(max_total: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read from data/insights.ndjson file."""
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
        pass
    return items


def _read_research_folder_fallback(max_total: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fallback: read directly from research ingest/ folder."""
    research_dir = os.path.join(os.path.dirname(__file__), 'research ingest')
    if not os.path.exists(research_dir):
        return []
    
    items = []
    try:
        # Get .txt and .json files
        pattern1 = os.path.join(research_dir, "*.txt")
        pattern2 = os.path.join(research_dir, "*.json")
        files = sorted(glob.glob(pattern1) + glob.glob(pattern2))
        
        for filepath in files:
            if max_total is not None and len(items) >= int(max_total):
                break
                
            try:
                # Extract title from filename
                basename = os.path.basename(filepath)
                title = os.path.splitext(basename)[0]
                if " - Ep. " in title:
                    parts = title.split(" - Ep. ", 1)
                    if len(parts) == 2:
                        title = f"Episode {parts[1]} ({parts[0]})"
                
                # Read content
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Create snippet
                snippet = content[:500] + "..." if len(content) > 500 else content
                
                item = {
                    "title": title,
                    "snippet": snippet,
                    "url": f"file://{os.path.abspath(filepath)}",
                    "source": "ram_dass_research_fallback"
                }
                items.append(item)
                
            except Exception:
                # Skip files that can't be read
                continue
                
    except Exception:
        # If anything fails, return whatever we have
        pass
    
    return items


if __name__ == '__main__':
    # simple standalone smoke-run
    items = gather_ingest()
    print(f"Gathered {len(items)} items")
    if items:
        print("First item keys:", list(items[0].keys()))