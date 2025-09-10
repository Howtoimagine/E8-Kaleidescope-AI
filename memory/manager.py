"""
Memory Manager extracted from the E8Mind monolith.

This is a skeleton to restore modular imports and enable incremental porting.
Full implementation should be ported from `e8_mind_server_M18.7.py`:
- MemoryManager (~lines 2683-3162)

Dependencies leveraged from modularized components:
- GraphDB (memory.graph)
- KDTree + cosine utils (memory.index)
- NoveltyScorer, HopfieldModern, KanervaSDM, VSA, MicroReranker, EmergenceSeed (memory.structures / core.data_structures)
"""

from __future__ import annotations
from typing import Any, List, Tuple, Optional, Iterable, Dict
import numpy as np

from .graph import GraphDB
from .index import KDTree, cosine_distances, cosine_similarity
from .structures import (
    NoveltyScorer, HopfieldModern, KanervaSDM,
    VSA, MicroReranker, EmergenceSeed
)

# Local helpers used by consolidation/synthesis routines
def _safe_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return np.zeros_like(v)
    return (v / (n + 1e-12)).astype(np.float32)

def _ensure_normalized(v: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    if arr.shape[0] != int(dim):
        if arr.shape[0] > int(dim):
            arr = arr[:int(dim)]
        else:
            arr = np.pad(arr, (0, int(dim) - arr.shape[0]))
    return _safe_normalize(arr)

class MemoryManager:
    """
    Skeleton of the MemoryManager orchestrating memory operations:
    - Vector indexing and retrieval
    - Novelty scoring / coherence evaluation
    - Consolidation via Hopfield / SDM / VSA
    - Graph memory integration

    TODO: Port full logic from the monolith.
    """

    def __init__(self, mind_instance: Optional[Any] = None, embed_dim: int = 1536, seed: int = 42):
        """Create a MemoryManager.

        Backward compatibility:
        The monolith originally constructed as MemoryManager(mind_instance, embed_dim=...).
        The modular skeleton initially only accepted embed_dim; this caused a TypeError
        when legacy code path expected the old signature. We now support both forms:

        - MemoryManager(mind_instance, embed_dim=1536)
        - MemoryManager(embed_dim=1536)  (mind_instance defaults to None)

        The mind instance (if provided) is stored as self.mind for any future
        cross-component calls that rely on accessing broader system context.
        """
        self.mind = mind_instance  # may be None
        self.embed_dim = int(embed_dim)
        self.seed = int(seed)
        # Core subsystems
        self.graph = GraphDB()
        # Backward compatibility alias: monolith references memory.graph_db
        # so expose graph_db with same object to avoid AttributeError.
        self.graph_db = self.graph  # type: ignore[attr-defined]
        self.novelty = NoveltyScorer(embed_dim=self.embed_dim, seed=self.seed)
        self.hopfield = HopfieldModern()
        self.sdm = KanervaSDM(dim=self.embed_dim, seed=self.seed)
        self.vsa = VSA(dim=self.embed_dim, seed=self.seed)
        self.reranker = MicroReranker()

        # Vector index (built on demand)
        self._index: Optional[KDTree] = None
        self._vectors: Optional[np.ndarray] = None
        self._ids: Optional[List[str]] = None

    # ------------------------------
    # Index Management
    # ------------------------------
    def build_index(self, ids: List[str], vectors: np.ndarray) -> None:
        """
        Build or rebuild the KDTree/FAISS-backed index.
        """
        X = np.asarray(vectors, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.embed_dim:
            raise ValueError(f"Expected (n, {self.embed_dim}) embeddings, got {X.shape}")
        if len(ids) != X.shape[0]:
            raise ValueError("ids length must match number of vectors")

        self._ids = list(ids)
        self._vectors = X
        self._index = KDTree(X)

    def is_index_ready(self) -> bool:
        return self._index is not None and self._vectors is not None and self._ids is not None

    # ------------------------------
    # Retrieval
    # ------------------------------
    def query(self, q: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (distances, indices) for the k nearest neighbors.
        """
        if not self.is_index_ready():
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        d, i = self._index.query(q, k=int(k))  # type: ignore[union-attr]
        return d, i

    def query_with_ids(self, q: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Convenience: returns [(id, distance), ...] sorted by distance.
        """
        d, i = self.query(q, k=k)
        if d.size == 0:
            return []
        # Handle both single and batch queries by flattening if needed
        if d.ndim > 1:
            d = d[0]
            i = i[0]
        assert self._ids is not None
        results = [(self._ids[int(ii)], float(dd)) for dd, ii in zip(d, i)]
        return results

    # ------------------------------
    # Novelty / Coherence Scoring
    # ------------------------------
    def score_novelty(self, vector: np.ndarray, k_context: int = 16) -> float:
        """
        Compute novelty of a vector given local neighborhood.
        """
        if not self.is_index_ready() or self._vectors is None:
            return 0.0
        _, idx = self.query(vector, k=min(k_context, self._vectors.shape[0]))
        if idx.size == 0:
            return 0.0
        if idx.ndim > 1:
            idx = idx[0]
        neighbors = self._vectors[idx]
        return float(self.novelty.score(vector, neighbors))  # type: ignore[attr-defined]

    # ------------------------------
    # Consolidation (Skeleton)
    # ------------------------------
    def consolidate(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply clean-up / consolidation pipeline (Hopfield, SDM, VSA, etc.).

        Enhanced behavior:
         - Build prototype ratings from similarity to centroid and update Hopfield prototypes.
         - Apply Hopfield clean_up per vector.
         - SDM write/read pass for refinement.
         - Return l2-normalized consolidated vectors.
        """
        X = np.asarray(vectors, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n, d = X.shape
        if n == 0:
            return X

        # Compute centroid and per-vector "ratings" as cosine similarity to centroid
        centroid = np.mean(X, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        ratings = (Xn @ centroid).astype(np.float32)

        # Update Hopfield prototypes using top-k rated vectors from this batch
        try:
            top_k = min(32, n)
            self.hopfield.update_prototypes(list(X), list(ratings), top_k=top_k)
        except Exception:
            # Best-effort: ignore prototype update failures
            pass

        # Step 1: Hopfield clean-up (per-vector)
        cleaned = []
        for v in X:
            try:
                v_clean = self.hopfield.clean_up(v, steps=3, tau=0.1)
            except Exception:
                v_clean = _safe_normalize(v)
            cleaned.append(v_clean)
        Xc = np.stack(cleaned, axis=0)

        # Step 2: SDM pass (write then read for slight error-correction)
        out = np.zeros_like(Xc)
        for i, v in enumerate(Xc):
            try:
                # write the cleaned vector into SDM and read back a refined version
                self.sdm.write(v, v)
                out_vec = self.sdm.read(v, k=8)
                out[i] = out_vec
            except Exception:
                out[i] = Xc[i]

        # Normalize output
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        out = out / norms
        return out

    # ------------------------------
    # Emergence Seed Handling (Skeleton)
    # ------------------------------
    async def synthesize_remnant(self, cluster: List[str], label_hint: Optional[str] = None
                                , llm_client: Optional[Any] = None, is_macro: bool = False
                                ) -> Tuple[str, np.ndarray, float]:
        """
        Create a remnant from a cluster of ids and return (remnant_id, embedding_vector, mass).

        Algorithm:
         - Gather vectors and node metadata (temperature, rating) for cluster members.
         - Compute temperature-weighted centroid and a weighted average rating.
         - Mass = sum(temperatures) * avg_rating (fallback heuristics applied).
         - Build Hopfield prototypes from top-rated members and clean the centroid.
         - If an llm_client is provided, request a JSON label/metaphor payload; otherwise
           fallback to deterministic label derived from label_hint.
        """
        vecs: List[np.ndarray] = []
        temps: List[float] = []
        ratings: List[float] = []
        present_ids: List[str] = []

        # Gather vectors and metadata
        for cid in cluster:
            v = self.get_vector_by_id(cid)
            node = self.graph.get_node(cid)
            if v is None:
                continue
            present_ids.append(cid)
            vecs.append(np.asarray(v, dtype=np.float32).reshape(-1))

            # temperature and rating metadata fallbacks
            temp = 1.0
            rating = 0.5
            try:
                if isinstance(node, dict):
                    temp = float(node.get("temperature", node.get("temp", 1.0)))
                    rating = float(node.get("rating", node.get("score", rating)))
            except Exception:
                temp = temp
                rating = rating
            temps.append(max(0.0, float(temp)))
            ratings.append(float(rating))

        d = int(self.embed_dim)
        if len(vecs) == 0:
            remnant_id = f"remnant:{label_hint or 'empty'}:0"
            return remnant_id, np.zeros((d,), dtype=np.float32), 0.0

        M = np.stack(vecs, axis=0)  # (n, d)
        temps_arr = np.asarray(temps, dtype=np.float32)
        if temps_arr.sum() <= 0:
            weights = np.ones((temps_arr.shape[0],), dtype=np.float32) / float(temps_arr.shape[0])
        else:
            weights = temps_arr / (temps_arr.sum() + 1e-12)

        # Weighted centroid
        centroid = (weights.reshape(-1, 1) * M).sum(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        # Weighted average rating
        avg_rating = float(np.dot(weights, np.asarray(ratings, dtype=np.float32)))

        # Mass heuristic
        mass = float(max(0.0, temps_arr.sum())) * max(0.0, avg_rating)

        # Build Hopfield prototypes from top-rated members (local batch)
        try:
            top_k = min(32, len(vecs))
            self.hopfield.update_prototypes(vecs, ratings, top_k=top_k)
        except Exception:
            pass

        # Clean centroid with Hopfield
        try:
            cleaned = self.hopfield.clean_up(centroid, steps=3, tau=0.1)
            cleaned_vec = _ensure_normalized(cleaned, d)
        except Exception:
            cleaned_vec = _ensure_normalized(centroid, d)

        # Optionally consult LLM for label/metaphor structured JSON
        label = (label_hint or "remnant").strip()
        metaphor = ""
        if llm_client is not None:
            try:
                prompt = {
                    "task": "label_and_metaphor",
                    "ids": present_ids,
                    "hint": label_hint,
                    "is_macro": bool(is_macro)
                }
                # try async generator patterns
                if hasattr(llm_client, "generate_json"):
                    out = await llm_client.generate_json(prompt)
                    if isinstance(out, dict):
                        label = str(out.get("label", label))
                        metaphor = str(out.get("metaphor", ""))
                else:
                    # generic generate
                    out = await llm_client.generate(prompt)  # type: ignore
                    if isinstance(out, dict):
                        label = str(out.get("label", label))
                        metaphor = str(out.get("metaphor", ""))
                    elif isinstance(out, str):
                        # try parse JSON
                        try:
                            import json
                            parsed = json.loads(out)
                            if isinstance(parsed, dict):
                                label = str(parsed.get("label", label))
                                metaphor = str(parsed.get("metaphor", ""))
                        except Exception:
                            pass
            except Exception:
                pass

        # Deterministic id from label hint and cluster membership (salted)
        salt = abs(hash((label, tuple(sorted(present_ids))))) % 100000
        remnant_id = f"remnant:{label}:{salt}"

        return remnant_id, cleaned_vec.astype(np.float32), float(mass)

    def insert_seed(self, seed: EmergenceSeed) -> None:
        """
        Insert an EmergenceSeed node into the graph DB with basic attributes.
        """
        self.graph.add_node(seed.remnant_id,
                            type="emergence_seed",
                            mass=float(seed.mass),
                            step_created=int(seed.step_created))

    # ------------------------------
    # Utility
    # ------------------------------
    def get_vector_by_id(self, item_id: str) -> Optional[np.ndarray]:
        """
        Return vector for an id if present in the in-memory store.
        """
        if self._ids is None or self._vectors is None:
            return None
        try:
            idx = self._ids.index(item_id)
        except ValueError:
            return None
        return self._vectors[idx]

__all__ = ["MemoryManager"]
