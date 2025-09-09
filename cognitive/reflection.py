from __future__ import annotations

from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque
import time
import json

class ReflectionEngine:
    """
    Lightweight reflection/meta-cognition logger.
    - Keeps a bounded history of reflection events
    - Emits simple console logs when available
    """

    def __init__(self, console: Optional[Any] = None, max_events: int = 256) -> None:
        self.console = console
        self.max_events = int(max_events)
        self.history: Deque[Dict[str, Any]] = deque(maxlen=self.max_events)

    def reflect(self, context: Dict[str, Any], tag: str = "subconscious") -> None:
        """
        Record a reflection event with timestamp and optional tag.
        """
        evt = {
            "ts": time.time(),
            "tag": str(tag),
            "context": context,
        }
        self.history.append(evt)
        # Best-effort console log
        try:
            if self.console:
                self.console.log("[Reflection]", json.dumps(evt))
        except Exception:
            pass

    def recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the most recent n reflection events (newest last)."""
        n = max(1, int(n))
        return list(self.history)[-n:]

    def summary(self, n: int = 10) -> Dict[str, Any]:
        """
        Simple summary over the last n events.
        """
        events = self.recent(n=n)
        return {
            "count": len(events),
            "first_ts": events[0]["ts"] if events else None,
            "last_ts": events[-1]["ts"] if events else None,
            "tags": list({e.get("tag", "unknown") for e in events}),
        }
