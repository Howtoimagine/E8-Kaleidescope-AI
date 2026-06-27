import os
from typing import List, Optional
from narrative_events import NarrEvent

GATE_LSS_DELTA = float(os.getenv("E8_NAR_GATE_LSS_DELTA", "0.08"))
MIN_STEPS_BETWEEN = int(os.getenv("E8_NAR_MIN_STEPS", "20"))


class NarrativeGate:
    """Decide when the subconscious should emit deterministic narrative."""

    def __init__(self) -> None:
        self.last_spoke_step = -10**9
        self.last_world_lss: Optional[float] = None

    def should_emit(
        self,
        step: int,
        lss_world_recent: Optional[float],
        events_since: List[NarrEvent],
    ) -> bool:
        if step - self.last_spoke_step < MIN_STEPS_BETWEEN:
            self.last_world_lss = lss_world_recent
            return False

        lock_hit = any(e.k == "ray_lock" for e in events_since)
        validator_flip = any(e.k == "validator" for e in events_since)
        phase_boundary = any(e.k == "phase" for e in events_since)

        lss_jump = False
        if lss_world_recent is not None and self.last_world_lss is not None:
            lss_jump = abs(lss_world_recent - self.last_world_lss) >= GATE_LSS_DELTA

        verdict = lock_hit or validator_flip or phase_boundary or lss_jump
        if verdict:
            self.last_spoke_step = step
        self.last_world_lss = lss_world_recent
        return verdict
