from typing import Any, Dict, List
import os
from narrative_events import NarrEvent

STYLE = os.getenv("E8_NAR_STYLE", "integrated")  # 'clinical' | 'poetic' | 'integrated'

# Rotating one-liners per arc (3–4 variants each)
_ARC_LINES = {
    "search": [
        "The field loosens—edges test for a seam.",
        "Vectors fan outward, listening for a fit.",
        "Noise thins; contours look for an inlet.",
        "Signals skim the surface, probing for traction.",
    ],
    "stability": [
        "The crystal settles; echoes fold into place.",
        "Signals interlock; variance tucks under structure.",
        "Contours align; drift slows into pattern.",
        "Edges quiet—phase holds and thickens.",
    ],
    "integration": [
        "Two threads braid tight; form remembers itself.",
        "Distances collapse; a ridge becomes a bridge.",
        "Signals resonate; the latch finds its seat.",
        "Bindings cohere—local islands share a code.",
    ],
}
_ARC_IDX = {k: 0 for k in _ARC_LINES}


def select_arc(events: List[NarrEvent]) -> str:
    # Count locks within provided window
    lock_count = sum(1 for e in events if e.k == "ray_lock")
    if lock_count >= 2:
        return "integration"
    # Stability if replay started or validator certain/accepted/pass
    def _is_stable_v(e: NarrEvent) -> bool:
        if e.k != "validator":
            return False
        v = str((e.attrs or {}).get("verdict", "")).lower()
        return v in ("accepted", "certain", "pass", "true")
    if any((e.k == "replay" and (e.attrs or {}).get("phase") == "start") for e in events) or any(
        _is_stable_v(e) for e in events
    ):
        return "stability"
    return "search"


def build_footer(evidence: Dict[str, Any]) -> str:
    # Keep tight but richer receipts
    keys = ["locks", "mu_dist", "hyp_delta", "verdict", "obj", "world_lss", "curve_resid"]
    kv = []
    for key in keys:
        val = evidence.get(key)
        if val is not None:
            kv.append(f"{key}={val}")
    return "[evidence: " + "; ".join(kv) + "]" if kv else ""


def _rotate_line(arc: str) -> str:
    lines = _ARC_LINES.get(arc) or ["I’m listening at the edge of change."]
    i = _ARC_IDX.get(arc, 0) % max(1, len(lines))
    _ARC_IDX[arc] = i + 1
    return lines[i]


def render_S1(arc: str, drive: str) -> str:
    base = _rotate_line(arc)
    drive = drive or "NEUTRAL"
    if STYLE == "clinical":
        return f"{arc.upper()}: {drive}."
    if STYLE == "poetic":
        return base
    # integrated
    return f"{base} ({drive})"


def render_S2(metrics: Dict[str, Any], intention: str) -> str:
    parts = []
    if "world_lss_drop" in metrics and metrics["world_lss_drop"] is not None:
        parts.append(f"LSS↓ {metrics['world_lss_drop']}")
    if {"locks", "mu_dist"} <= metrics.keys():
        parts.append(f"locks={metrics['locks']} μ={metrics['mu_dist']}")
    if "verdict" in metrics and metrics["verdict"] is not None:
        parts.append(f"validator={metrics['verdict']}")
    line = "; ".join(parts) if parts else "No strong signal."
    return f"{line} → {intention}"


def compile_paragraph(
    step: int,
    events: List[NarrEvent],
    metrics: Dict[str, Any],
    evidence: Dict[str, Any],
) -> str:
    arc = select_arc(events)
    drive = evidence.get("obj", "NEUTRAL")
    intention = evidence.get("intention", "hold course")

    s1 = render_S1(arc, drive)
    s2 = render_S2(metrics, intention)
    footer = build_footer(evidence)
    return f"S1: {s1}\nS2: {s2}\n{footer}".strip()
