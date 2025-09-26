"""
Core Data Structures for the E8Mind

This module contains fundamental data structures used throughout the E8Mind system,
including emergent seeds, task management, market data, and memory structures.
"""

import os
import time
import logging
import collections
import threading
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Import constants from environment
GLOBAL_SEED = int(os.getenv("E8_SEED", "42"))
EMBED_DIM = int(os.getenv("E8_EMBED_DIM", "1536"))

# Optional imports with fallbacks
try:
    from sklearn.neighbors import KDTree as _SKKDTree
except Exception:
    _SKKDTree = None

try:
    from scipy.spatial import KDTree as _SPKDTree
except Exception:
    _SPKDTree = None

try:
    import faiss
    _FAISS = True
except Exception:
    _FAISS = False

try:
    import networkx as nx
    from networkx.readwrite import json_graph
except Exception:
    nx = None
    class _JG:
        def node_link_data(self, g): return {"nodes": [], "links": []}
        def node_link_graph(self, d): return None
    json_graph = _JG()


@dataclass
class EmergenceSeed:
    """Represents a black hole event remnant in the E8Mind memory system."""
    remnant_id: str
    embedding_vector: np.ndarray
    projected_vector: np.ndarray
    mass: float
    absorbed_ids: List[str]
    step_created: int


@dataclass
class AutoTask:
    """Represents an automatically generated task for the curriculum system."""
    id: str
    label: str
    reason: str
    novelty: float
    coherence: float
    status: str = "pending"
    created_step: int = 0


class Bar:
    """Represents market bar data (OHLC)."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class DecodeState:
    """State information for the holographic decoder."""
    current_idx: int
    shadow_ids: np.ndarray
    slice_id: int
    seen_tokens: set
    emap: Any  # EntropyMap reference
    holo: Any  # HoloEncoder reference


class UniversalEmbeddingAdapter:
    """Dimension-agnostic adapter with whitening and online calibration."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        calibration_capacity: int = 256,
        min_pairs_to_fit: int = 8,
        ridge_lambda: float = 1e-3,
        blend: float = 0.5,
        cooldown_seconds: float = 3.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self._rng = np.random.default_rng(GLOBAL_SEED)
        self._norm_epsilon = 1e-6
        self._logger = logger or logging.getLogger("UniversalEmbeddingAdapter")

        self.W = self._init_projection_matrix()

        # Running normalization stats
        self._input_mean = np.zeros(self.in_dim, dtype=np.float32)
        self._input_m2 = np.zeros(self.in_dim, dtype=np.float32)
        self._input_count = 0

        self._calibration_pairs = collections.deque(maxlen=max(8, int(calibration_capacity)))
        self._min_pairs_to_fit = max(4, int(min_pairs_to_fit))
        self._ridge_lambda = float(ridge_lambda)
        self._blend = float(np.clip(blend, 0.0, 1.0))
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._last_fit_ts = 0.0
        self._fit_count = 0
        self._stats = {"pairs_registered": 0}

        self._lock = threading.RLock()

    def _init_projection_matrix(self) -> np.ndarray:
        if self.in_dim == self.out_dim:
            return np.eye(self.in_dim, dtype=np.float32)

        mat = self._rng.standard_normal((self.in_dim, self.out_dim)).astype(np.float32)
        try:
            if self.in_dim >= self.out_dim:
                q, _ = np.linalg.qr(mat)
                mat = q[:, :self.out_dim]
        except Exception:
            pass

        col_norms = np.linalg.norm(mat, axis=0, keepdims=True)
        mat = np.divide(mat, np.maximum(col_norms, self._norm_epsilon), out=mat)
        return mat.astype(np.float32)

    def _prepare_input(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.in_dim:
            padded = np.zeros(self.in_dim, dtype=np.float32)
            size = min(arr.shape[0], self.in_dim)
            padded[:size] = arr[:size]
            arr = padded
        return arr

    def _prepare_target(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.out_dim:
            padded = np.zeros(self.out_dim, dtype=np.float32)
            size = min(arr.shape[0], self.out_dim)
            padded[:size] = arr[:size]
            arr = padded
        return arr

    def _normalize_input(self, vector: np.ndarray) -> np.ndarray:
        if self._input_count < 10:
            return vector
        variance = self._input_m2 / max(self._input_count - 1, 1.0)
        std = np.sqrt(np.maximum(variance, self._norm_epsilon))
        return (vector - self._input_mean) / std

    def _update_input_stats(self, vector: np.ndarray) -> None:
        self._input_count += 1
        delta = vector - self._input_mean
        self._input_mean += delta / float(self._input_count)
        delta2 = vector - self._input_mean
        self._input_m2 += delta * delta2

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        src = self._prepare_input(vector)
        normalized = self._normalize_input(src)
        with self._lock:
            W = self.W.copy()
        projected = normalized @ W
        norm = float(np.linalg.norm(projected))
        if norm > self._norm_epsilon:
            projected = projected / norm
        self._update_input_stats(src)
        return projected.astype(np.float32)

    def register_alignment_pair(self, source_vec: np.ndarray, target_vec: np.ndarray, weight: float = 1.0) -> None:
        src = self._prepare_input(source_vec)
        tgt = self._prepare_target(target_vec)
        w = float(max(weight, 1e-6))

        with self._lock:
            self._calibration_pairs.append((src, tgt, w))
            self._stats["pairs_registered"] += 1
            now = time.monotonic()
            if len(self._calibration_pairs) >= self._min_pairs_to_fit and (now - self._last_fit_ts) >= self._cooldown_seconds:
                self._recompute_projection(now)

    def _recompute_projection(self, timestamp: Optional[float] = None) -> bool:
        pairs = list(self._calibration_pairs)
        if not pairs:
            return False

        sources = np.stack([self._normalize_input(p[0]) for p in pairs], dtype=np.float32)
        targets = np.stack([self._normalize_target(p[1]) for p in pairs], dtype=np.float32)
        weights = np.sqrt(np.array([p[2] for p in pairs], dtype=np.float32))

        sources *= weights[:, None]
        targets *= weights[:, None]

        try:
            W_fit, *_ = np.linalg.lstsq(sources, targets, rcond=self._ridge_lambda)
        except Exception as exc:
            self._log(f"calibration failed: {exc}")
            return False

        if not np.all(np.isfinite(W_fit)):
            self._log("calibration produced non-finite weights; rejecting update")
            return False

        W_fit = W_fit.astype(np.float32)
        if self._blend > 0.0:
            self.W = (1.0 - self._blend) * self.W + self._blend * W_fit
        else:
            self.W = W_fit

        col_norms = np.linalg.norm(self.W, axis=0, keepdims=True)
        self.W = np.divide(self.W, np.maximum(col_norms, self._norm_epsilon), out=self.W)

        self._last_fit_ts = timestamp if timestamp is not None else time.monotonic()
        self._fit_count += 1
        self._log(f"recalibrated on {len(pairs)} pairs (fit #{self._fit_count})")
        return True

    def _normalize_target(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm > self._norm_epsilon:
            return vector / norm
        return vector

    def _log(self, message: str) -> None:
        try:
            if self._logger is not None:
                self._logger.debug(f"[UniversalAdapter] {message}")
        except Exception:
            pass


class KDTree:
    """
    A wrapper for scikit-learn/scipy KDTree with optional FAISS and a NumPy fallback.
    """
    def __init__(self, data):
        X = np.asarray(data, dtype=np.float32)
        
        if '_FAISS' in globals() and _FAISS and X.ndim == 2 and X.size:
            self._is_faiss = True
            self._dim = X.shape[1]
            self._faiss_index = faiss.IndexFlatIP(self._dim)
            # normalize for cosine similarity via dot
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            Xn = X / norms
            self._faiss_index.add(Xn)  # type: ignore[attr-defined]
            self.n = X.shape[0]
            self._is_fallback = False
            self._impl = None
        elif _SKKDTree is not None:
            self._impl = _SKKDTree(X)
            self.n = self._impl.data.shape[0]
            self._is_fallback = False
            self._is_faiss = False
        elif _SPKDTree is not None:
            self._impl = _SPKDTree(X)
            self.n = self._impl.n
            self._is_fallback = False
            self._is_faiss = False
        else:
            self._impl = X
            self.n = self._impl.shape[0]
            self._is_fallback = True
            self._is_faiss = False

    def query(self, q, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Query for k nearest neighbors."""
        t0 = time.perf_counter()
        q_arr = np.asarray(q, dtype=np.float32)
        is_single_query = q_arr.ndim == 1
        q_2d = np.atleast_2d(q_arr)

        if getattr(self, '_is_faiss', False):
            q2 = q_2d
            norms = np.linalg.norm(q2, axis=1, keepdims=True) + 1e-12
            qn = q2 / norms
            try:
                qfaiss = qn.astype(np.float32)
                if qfaiss.ndim == 1:
                    qfaiss = qfaiss.reshape(1, -1)
                D, I = self._faiss_index.search(qfaiss, int(k))  # type: ignore[attr-defined]
            except Exception:
                # Fallback: no results
                D = np.ones((q_2d.shape[0], int(k)), dtype=np.float32)
                I = -np.ones((q_2d.shape[0], int(k)), dtype=np.int64)
            # Convert cosine sim to distance
            d = 1.0 - D
            i = I
        elif not self._is_fallback and hasattr(self._impl, 'query'):
            d, i = self._impl.query(q_2d, k=k)  # type: ignore[call-arg]
            d = np.asarray(d, dtype=np.float32)
            i = np.asarray(i, dtype=np.int64)
        else:
            # NumPy fallback for both single and batch queries
            all_dists = []
            all_indices = []
            data_points = self._impl
            
            for query_vector in q_2d:
                # Calculate Euclidean distances from the current query vector to all data points
                distances = np.sqrt(np.sum((data_points - query_vector)**2, axis=1))
                
                # Get the indices of the k smallest distances
                if k < self.n:
                    # Find the k nearest indices (unsorted)
                    nearest_idx = np.argpartition(distances, k-1)[:k]
                    # Sort only that small partition by distance to get the correct order
                    sorted_partition_indices = np.argsort(distances[nearest_idx])
                    idx = nearest_idx[sorted_partition_indices]
                else:
                    # If k is as large as the dataset, just sort everything
                    idx = np.argsort(distances)[:k]

                all_indices.append(idx)
                all_dists.append(distances[idx])
            
            d = np.array(all_dists, dtype=np.float32)
            i = np.array(all_indices, dtype=np.int64)

        # Return results in the expected shape
        d_out = d.ravel() if is_single_query else d
        i_out = i.ravel() if is_single_query else i
        
        # Track latency rolling average
        try:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if not hasattr(self, '_latency_ms'):
                self._latency_ms = []
            self._latency_ms.append(dt_ms)
            if len(self._latency_ms) > 128:
                self._latency_ms.pop(0)
        except Exception:
            pass
        
        return d_out, i_out


class GraphDB:
    """A graph database wrapper around NetworkX for managing conceptual relationships."""
    
    def __init__(self):
        if nx is None:
            raise ImportError("networkx library is required for GraphDB.")
        self.graph = nx.Graph()

    def add_node(self, node_id: str, **attrs):
        """Adds a node to the graph with the given attributes."""
        self.graph.add_node(node_id, **attrs)

    def add_edge(self, source_id: str, target_id: str, **attrs):
        """Adds an edge between two nodes with the given attributes."""
        self.graph.add_edge(source_id, target_id, **attrs)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a node's data."""
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]
        return None

    def get_neighbors(self, node_id: str) -> List[str]:
        """Gets the neighbors of a node."""
        if self.graph.has_node(node_id):
            return list(self.graph.neighbors(node_id))
        return []

    def compute_and_store_communities(self, partition_key: str = "community_id"):
        """Computes Louvain communities and stores the partition ID on each node."""
        try:
            from networkx.algorithms import community as nx_comm
        except Exception:
            return
            
        if self.graph.number_of_nodes() < 10:
            return
            
        try:
            communities_iter = nx_comm.louvain_communities(self.graph, seed=GLOBAL_SEED)
            communities = list(communities_iter)
            for i, community_nodes in enumerate(communities):
                for node_id in community_nodes:
                    if self.graph.has_node(node_id):
                        self.graph.nodes[node_id][partition_key] = i
        except Exception as e:
            print(f"[GraphDB] Community detection failed: {e}")

    def increment_edge_weight(self, u: str, v: str, delta: float = 0.1, 
                            min_w: float = 0.0, max_w: float = 10.0, **attrs):
        """Create edge if absent; add delta to 'weight' clamped to [min_w, max_w]."""
        try:
            if not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v, weight=max(min_w, delta), **attrs)
            else:
                current_weight = self.graph.get_edge_data(u, v, default={'weight': 0.0}).get('weight', 0.0)
                new_weight = float(current_weight) + float(delta)
                new_weight = min(max_w, max(min_w, new_weight))
                self.graph[u][v]['weight'] = new_weight
                for k, val in attrs.items():
                    self.graph[u][v][k] = val
        except Exception as e:
            print(f"[GraphDB] increment_edge_weight failed: {e}")
