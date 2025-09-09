from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from core.data_structures import EmergenceSeed
from physics.engines import QuantumEngine, ClassicalEngine
from physics.e8_lattice import E8Physics
from memory.manager import MemoryManager
from .emergence import detect_emergence_minima
from .reflection import ReflectionEngine

@dataclass
class SubconsciousConfig:
    """Lightweight config for the SubconsciousLayer."""
    emergence_top_k: int = 3
    emergence_sigma_threshold: float = 2.0  # z-score below mean
    curiosity_alpha: float = 0.0           # default off
    weyl_draws: int = 0                    # default off

class SubconsciousLayer:
    """
    Orchestrates background consolidation and emergence detection.
    Integrates Physics, Engines, and MemoryManager. Keeps logic lightweight.
    """

    def __init__(
        self,
        physics: E8Physics,
        qengine: QuantumEngine,
        cengine: ClassicalEngine,
        memory: MemoryManager,
        console: Optional[Any] = None,
        config: Optional[SubconsciousConfig] = None,
    ) -> None:
        self.physics = physics
        self.qengine = qengine
        self.cengine = cengine
        self.memory = memory
        self.console = console
        self.config = config or SubconsciousConfig()
        self.reflector = ReflectionEngine(console=console)

        self.step_counter: int = 0
        self.last_potential: Optional[np.ndarray] = None
        self.last_emergence: List[Tuple[int, float]] = []

    def log(self, *args):
        try:
            if self.console:
                self.console.log("[Subconscious]", *args)
        except Exception:
            pass

    def tick(
        self,
        anchors: Optional[List[Tuple[np.ndarray, float]]] = None,
        curiosity_visits: Optional[np.ndarray] = None,
        label_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Single subconscious step:
        - Optionally updates anchors
        - Computes/updates potential field
        - Detects emergence minima
        - Optionally synthesizes and inserts EmergenceSeeds into memory graph
        Returns a summary dict.
        """
        self.step_counter += 1

        # 1) Potential update (keep lightweight and safe if no anchors)
        try:
            if anchors is not None:
                self.qengine.set_anchors(anchors)
            V = self.qengine.potential(
                curiosity_visits=curiosity_visits,
                curiosity_alpha=float(self.config.curiosity_alpha),
                weyl_draws=int(self.config.weyl_draws),
                weyl_seed=None,
            )
            self.last_potential = V
        except Exception as e:
            self.log(f"potential_update_error: {e}")
            V = None

        # 2) Emergence detection (minima under sigma threshold)
        minima: List[Tuple[int, float]] = []
        if V is not None and V.size > 0:
            minima = detect_emergence_minima(
                V,
                top_k=int(self.config.emergence_top_k),
                sigma_threshold=float(self.config.emergence_sigma_threshold),
            )
            self.last_emergence = minima

        created: List[str] = []
        # 3) Seed synthesis (heuristic, non-blocking path)
        # Use ClassicalEngine blueprint projection matrix if available for projected_vector
        if minima:
            for idx, val in minima:
                try:
                    vec8 = self.physics.roots_unit[idx].astype(np.float32)
                    # projected vector: reuse 8D for now (blueprint projection is 3D, EmergenceSeed expects vector)
                    projected_vec = vec8.copy()
                    remnant_id = f"remnant:e8root:{idx}"
                    seed = EmergenceSeed(
                        remnant_id=remnant_id,
                        embedding_vector=vec8,
                        projected_vector=projected_vec,
                        mass=float(-val),  # lower potential => higher mass
                        absorbed_ids=[f"root:{idx}"],
                        step_created=int(self.step_counter),
                    )
                    # Insert minimal seed node in graph
                    self.memory.insert_seed(seed)
                    created.append(remnant_id)
                except Exception as e:
                    self.log(f"seed_insert_error idx={idx}: {e}")

        # 4) Reflection log
        try:
            self.reflector.reflect(
                context={
                    "step": self.step_counter,
                    "minima": minima,
                    "created": created,
                    "anchors": len(anchors) if anchors is not None else 0,
                }
            )
        except Exception:
            pass

        return {
            "step": self.step_counter,
            "created": created,
            "minima": minima,
            "potential_stats": (
                {
                    "min": float(np.min(V)),
                    "max": float(np.max(V)),
                    "mean": float(np.mean(V)),
                    "std": float(np.std(V)),
                }
                if V is not None and V.size > 0
                else None
            ),
        }
