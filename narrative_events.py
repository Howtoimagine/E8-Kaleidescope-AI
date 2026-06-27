from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List
import os
import time

MAX_EVENTS = int(os.getenv("E8_EVT_MAX", "256"))


@dataclass
class NarrEvent:
    t: int
    k: str
    ids: Dict[str, Any] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class EventKernel:
    """Lightweight rolling buffer for narrative-critical events."""

    def __init__(self, maxlen: int = MAX_EVENTS):
        self.buf: Deque[NarrEvent] = deque(maxlen=maxlen)

    def record(
        self,
        t: int,
        kind: str,
        ids: Dict[str, Any] | None = None,
        attrs: Dict[str, Any] | None = None,
    ) -> None:
        self.buf.append(NarrEvent(t=int(t), k=kind, ids=ids or {}, attrs=attrs or {}))

    def recent(self, n: int = 64) -> List[NarrEvent]:
        return list(self.buf)[-n:]

    def find_since(self, t_min: int, kinds: List[str]) -> List[NarrEvent]:
        kind_set = set(kinds)
        return [e for e in self.buf if e.t >= t_min and e.k in kind_set]
