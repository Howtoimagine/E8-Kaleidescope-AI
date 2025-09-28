from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from core.data_structures import KDTree
except Exception:  # pragma: no cover
    KDTree = None  # type: ignore


@dataclass
class PathAsset:
    nodes: List[str]
    length: float
    meta: Dict[str, Any]


class ProximityEngine:
    """Lightweight proximity and tracing helpers for shells and memory.

    This modular version focuses on stable, dependency-light primitives:
    - Per-dimension shell indices built via KDTree (if available) or NumPy
    - KNN queries for vectors
    - Optional shortest-path on the memory GraphDB when available
    """

    def __init__(self, console: Any = None):
        self.console = console
        self._shell_index: Dict[int, Dict[str, Any]] = {}

    # --- Shell indexing ---
    def update_shell_index(self, dim: int, shell: Any):
        """Build or refresh a KNN index over shell vectors.

        Expects shell.get_all_vectors_as_matrix() -> (matrix (N,D), node_ids)
        """
        try:
            mat, node_ids = shell.get_all_vectors_as_matrix()
        except Exception as e:
            if self.console:
                try:
                    self.console.log(f"[Proximity] No shell vectors for d={dim}: {e}")
                except Exception:
                    pass
            self._shell_index.pop(int(dim), None)
            return

        if mat is None or node_ids is None or len(node_ids) == 0:
            self._shell_index.pop(int(dim), None)
            return

        index = None
        if KDTree is not None:
            try:
                index = KDTree(mat)
            except Exception as e:
                index = None
                if self.console:
                    try:
                        self.console.log(f"[Proximity] KDTree build failed (d={dim}): {e}")
                    except Exception:
                        pass
        self._shell_index[int(dim)] = {"index": index, "matrix": mat, "ids": node_ids}

    def _ensure_index(self, dim: int, shell: Any):
        if int(dim) not in self._shell_index:
            self.update_shell_index(dim, shell)
        return self._shell_index.get(int(dim))

    def knn(self, dim: int, shell: Any, query_vec: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Return up to k nearest node ids to the query vector."""
        entry = self._ensure_index(dim, shell)
        if not entry:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        mat = entry["matrix"]
        ids = entry["ids"]
        index = entry["index"]
        if index is not None:
            try:
                d, i = index.query(q, k=int(min(k, len(ids))))
                i = np.atleast_1d(i)
                d = np.atleast_1d(d)
                return [(ids[int(ii)], float(dd)) for ii, dd in zip(i, d)]
            except Exception:
                pass
        # NumPy fallback
        diffs = mat - q.reshape(1, -1)
        d2 = np.sum(diffs * diffs, axis=1)
        order = np.argsort(d2)[: max(1, int(k))]
        return [(ids[int(i)], float(np.sqrt(d2[int(i)]))) for i in order]

    # --- Graph path tracing ---
    def shortest_path(self, memory: Any, source_node: str, target_node: str) -> Optional[PathAsset]:
        """Shortest path on GraphDB if present; returns PathAsset or None."""
        G = getattr(memory, "graph_db", None)
        if G is None or getattr(G, "graph", None) is None:
            return None
        try:
            import networkx as nx  # local import to avoid hard dep in init
            path = nx.shortest_path(G.graph, source=str(source_node), target=str(target_node), weight="weight", method="dijkstra")
            length = 0.0
            for u, v in zip(path[:-1], path[1:]):
                w = float(G.graph[u][v].get("weight", 1.0))
                length += max(1e-6, 1.0 / (w + 1e-9))
            return PathAsset(nodes=[str(n) for n in path], length=float(length), meta={"algo": "dijkstra"})
        except Exception:
            return None
