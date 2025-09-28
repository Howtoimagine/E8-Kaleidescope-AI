from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Reuse the canonical GraphDB from core to avoid duplication
try:
    from core.data_structures import GraphDB as _CoreGraphDB
except Exception:  # pragma: no cover
    _CoreGraphDB = None  # type: ignore


GraphDB = _CoreGraphDB  # public alias


class GeometryHygieneMixin:
    """Small helpers for geometric hygiene and lattice snapping.

    These utilities are intentionally conservative and dependency-light.
    """

    _norm_eps: float = 1e-6

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(arr))
        if n < self._norm_eps:
            return arr
        return (arr / n).astype(np.float32)

    def _snap_to_lattice(self, v: np.ndarray, dim: int) -> np.ndarray:
        """Project vector to R^dim, pad/trim, and L2-normalize.

        Acts as a stable baseline when no learned projector is available.
        """
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.size != dim:
            out = np.zeros(dim, dtype=np.float32)
            n = min(arr.size, dim)
            out[:n] = arr[:n]
            arr = out
        return self._normalize(arr)


class MemoryManager(GeometryHygieneMixin):
    """Coordinates memory storage, geometric shells, and proximity indices.

    Minimal, dependency-light implementation that exposes a compatible surface
    with the legacy monolith.
    """

    def __init__(self, console: Any = None):
        self.console = console
        self.graph_db = GraphDB() if GraphDB is not None else None
        self.shells: Dict[int, Any] = {}  # dim -> DimensionalShell
        self.proximity = None  # set via attach_proximity
        # Main storage of vectors keyed by node id (for similarity lookups)
        self.main_vectors: Dict[str, np.ndarray] = {}

        # Optional learned projector/autoencoder attached by the Mind
        self.autoencoder = None

    # --- Shell management ---
    def ensure_shell(self, dim: int, mind_instance: Any) -> Any:
        from .shell import DimensionalShell  # lazy import to avoid cycles
        d = int(dim)
        sh = self.shells.get(d)
        if sh is None:
            sh = DimensionalShell(d, mind_instance)
            self.shells[d] = sh
            if self.console:
                try:
                    self.console.log(f"[Memory] Created DimensionalShell(d={d}).")
                except Exception:
                    pass
        return sh

    def attach_proximity(self, engine: Optional[Any] = None, *args, **kwargs) -> Any:
        """Attach or create a proximity engine.

        Compatibility: legacy code may pass (shell_dims, mind, console). We ignore
        positional args and just construct a default engine when none provided.
        """
        if engine is None:
            from .proximity import ProximityEngine  # lazy import
            console = kwargs.get("console", getattr(self, "console", None))
            engine = ProximityEngine(console=console)
        self.proximity = engine
        return self.proximity

    # --- Vector storage & similarity ---
    def add_vector_to_main_storage(self, node_id: str, vector: np.ndarray) -> None:
        v = self._normalize(np.asarray(vector, dtype=np.float32))
        self.main_vectors[str(node_id)] = v

    def find_similar_in_main_storage(self, query: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if not self.main_vectors:
            return []
        q = self._normalize(query)
        items = list(self.main_vectors.items())
        ids = [nid for nid, _ in items]
        mat = np.stack([vec for _, vec in items], axis=0)
        # cosine similarity
        sims = (mat @ q.reshape(-1))
        if not np.all(np.isfinite(sims)):
            sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.argsort(-sims)[: max(1, int(k))]
        return [(ids[i], float(sims[i])) for i in idx]

    # --- Projectors ---
    def project_to_dim(self, v: np.ndarray, d: int = 8) -> np.ndarray:
        """Project via attached autoencoder if available; else pad/trim + normalize."""
        try:
            if self.autoencoder is not None and getattr(self.autoencoder, "is_trained", False):
                # autoencoder may accept torch or numpy; it returns the same type
                out = self.autoencoder.project_to_dim(v, d)
                if hasattr(out, "detach"):
                    try:
                        import torch  # type: ignore
                        return out.detach().cpu().numpy().reshape(-1).astype(np.float32)
                    except Exception:
                        pass
                return np.asarray(out, dtype=np.float32).reshape(-1)
        except Exception:
            pass
        return self._snap_to_lattice(v, int(d))

    def project_to_dim8(self, v: np.ndarray) -> np.ndarray:
        return self.project_to_dim(v, 8)
