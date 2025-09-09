"""
Core Data Structures for the E8Mind

This module contains fundamental data structures used throughout the E8Mind system,
including emergent seeds, task management, market data, and memory structures.
"""

import os
import time
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
    """
    Adapts embeddings from one dimension to another using a learned transformation matrix.
    """
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if in_dim == out_dim:
            self.W = np.eye(in_dim, dtype=np.float32)
        else:
            rng = np.random.default_rng(GLOBAL_SEED)
            self.W = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
            self.W /= np.linalg.norm(self.W, axis=0, keepdims=True)

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        """Transform a vector from input dimension to output dimension."""
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        
        if vector.shape[0] != self.in_dim:
            # Pad or truncate if there's a mismatch
            padded_vec = np.zeros(self.in_dim, dtype=np.float32)
            size_to_copy = min(vector.shape[0], self.in_dim)
            padded_vec[:size_to_copy] = vector[:size_to_copy]
            vector = padded_vec
        
        return vector @ self.W


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
            self._faiss_index.add(Xn)
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
                D, I = self._faiss_index.search(qfaiss, int(k))
            except Exception:
                # Fallback: no results
                D = np.ones((q_2d.shape[0], int(k)), dtype=np.float32)
                I = -np.ones((q_2d.shape[0], int(k)), dtype=np.int64)
            # Convert cosine sim to distance
            d = 1.0 - D
            i = I
        elif not self._is_fallback and hasattr(self._impl, 'query'):
            d, i = self._impl.query(q_2d, k=k)
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
