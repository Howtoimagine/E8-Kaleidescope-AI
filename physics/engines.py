from __future__ import annotations
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np

from .e8_lattice import E8Physics, weyl_average_potential, add_curiosity_penalty
from .potential_fields import MultiAnchorField


class QuantumEngine:
    """
    Quantum-like engine that operates over the E8 lattice using potential fields.
    Provides smooth field composition (anchors), optional Weyl averaging, and curiosity penalties.
    """

    def __init__(self, physics: E8Physics, config: Optional[dict] = None, console: Any = None):
        self.physics = physics
        self.console = console
        self.config = config or {}
        kernel = self.config.get("kernel", "cosine")
        rbf_sigma = float(self.config.get("rbf_sigma", 0.8))
        self.field = MultiAnchorField(self.physics, kernel=kernel, rbf_sigma=rbf_sigma)
        self._mind = None
        self._last_potential: Optional[np.ndarray] = None
        self._last_anchors: List[Tuple[np.ndarray, float]] = []

    def attach_mind(self, mind_instance: Any):
        self._mind = mind_instance
        if self.console:
            try:
                self.console.log("[QuantumEngine] Mind attached.")
            except Exception:
                pass

    def set_anchors(self, anchors: Sequence[Tuple[np.ndarray, float]]) -> None:
        """Set anchor (vector_8d, weight) pairs. Vectors should be 8D and approximately unit length."""
        # Normalize anchors and store
        normed: List[Tuple[np.ndarray, float]] = []
        for vec, w in anchors:
            v = np.asarray(vec, dtype=np.float32).reshape(-1)
            if v.shape[0] != 8:
                raise ValueError(f"Anchor vector must be 8D, got {v.shape}")
            v = v / (np.linalg.norm(v) + 1e-12)
            normed.append((v, float(w)))
        self._last_anchors = normed
        self.field.set(normed)

    def potential(
        self,
        curiosity_visits: Optional[np.ndarray] = None,
        curiosity_alpha: float = 0.12,
        weyl_draws: int = 0,
        weyl_seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute current potential field over all 240 E8 roots.
        - If weyl_draws &gt; 0: average potential under random Weyl group transforms.
        - If curiosity_visits provided: add curiosity penalty.
        Returns array shape (240,) where lower is more attractive.
        """
        if weyl_draws and weyl_draws > 0:
            V = weyl_average_potential(self.physics, self.field, draws=int(weyl_draws), seed=weyl_seed)
        else:
            V = self.field.potential()

        if curiosity_visits is not None:
            V = add_curiosity_penalty(V, curiosity_visits, alpha=float(curiosity_alpha))

        self._last_potential = V.astype(np.float32)
        return self._last_potential

    def sample(self, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Return the indices and values of the top_k lowest potential lattice points.
        Requires potential() to have been called.
        """
        if self._last_potential is None:
            raise RuntimeError("Call potential() before sample().")
        V = self._last_potential
        k = max(1, min(int(top_k), V.shape[0]))
        idx = np.argpartition(V, k - 1)[:k]
        idx = idx[np.argsort(V[idx])]
        return [(int(i), float(V[i])) for i in idx]

    def step(
        self,
        anchors: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
        curiosity_visits: Optional[np.ndarray] = None,
        curiosity_alpha: float = 0.12,
        weyl_draws: int = 0,
        weyl_seed: Optional[int] = None,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Convenience one-shot: set anchors (if provided), compute potential, return best candidates.
        """
        if anchors is not None:
            self.set_anchors(anchors)
        self.potential(curiosity_visits=curiosity_visits, curiosity_alpha=curiosity_alpha,
                       weyl_draws=weyl_draws, weyl_seed=weyl_seed)
        return self.sample(top_k=top_k)


class ClassicalEngine:
    """
    Classical engine exposing deterministic utilities over the E8 lattice.
    Provides helpers for nearest root queries, heat masks, blueprint projection, and potential queries.
    """

    def __init__(self, physics: E8Physics, config: Optional[dict] = None, console: Any = None):
        self.physics = physics
        self.console = console
        self.config = config or {}

    def nearest_root(self, vector_8d: np.ndarray) -> Optional[int]:
        """Return index of nearest lattice root for a given 8D vector."""
        return self.physics.find_nearest_root_index(np.asarray(vector_8d, dtype=np.float32).reshape(-1))

    def heat_mask(self, center_idx: int, sigma: float = 1.25) -> np.ndarray:
        """Return heat diffusion mask around a center lattice index."""
        return self.physics.heat_mask_cached(int(center_idx), sigma=float(sigma))

    def blueprint(self, seed: Optional[int] = None):
        """Return a quasicrystal blueprint mapping for visualization or downstream geometry."""
        if seed is None:
            return self.physics.generate_quasicrystal_blueprint()
        return self.physics.generate_quasicrystal_blueprint(seed=int(seed))

    def potential_from_anchors(
        self,
        anchors: Sequence[Tuple[np.ndarray, float]],
        kernel: str = "cosine",
        rbf_sigma: float = 0.8,
        curiosity_visits: Optional[np.ndarray] = None,
        curiosity_alpha: float = 0.12,
    ) -> np.ndarray:
        """
        Compute potential field given anchors using a fresh MultiAnchorField (without internal state).
        """
        fld = MultiAnchorField(self.physics, kernel=kernel, rbf_sigma=float(rbf_sigma))
        normed: List[Tuple[np.ndarray, float]] = []
        for vec, w in anchors:
            v = np.asarray(vec, dtype=np.float32).reshape(-1)
            if v.shape[0] != 8:
                raise ValueError(f"Anchor vector must be 8D, got {v.shape}")
            v = v / (np.linalg.norm(v) + 1e-12)
            normed.append((v, float(w)))
        fld.set(normed)
        V = fld.potential()
        if curiosity_visits is not None:
            V = add_curiosity_penalty(V, curiosity_visits, alpha=float(curiosity_alpha))
        return V.astype(np.float32)


__all__ = ["QuantumEngine", "ClassicalEngine"]
