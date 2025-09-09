
# wavey/__init__.py — upgraded for E8 Mind Server (Pillars 1–5)
# Provides: WaveyE8Bridge, integrate_one_cycle, PotentialFunction
# - Pure NumPy, no external deps.
# - Outputs node_potentials (240,) to align Wavey attention with the graph-aware Hamiltonian.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

__all__ = ["WaveyE8Bridge", "integrate_one_cycle", "PotentialFunction"]

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class PotentialFunction:
    center: np.ndarray            # raw embedding in encoder space (e.g., 1536-d)
    depth: float                  # strength / “depth” of the potential well
    label: Optional[str] = None   # optional label (e.g., node id)

# -----------------------------
# Bridge
# -----------------------------

class WaveyE8Bridge:
    """
    Minimal adapter that scores memory items and proposes potentials + attention.
    Strategy (fast & deterministic):
      - Build novelty scores as distance to the global centroid.
      - Take top-K items as potential centers.
      - Normalize scores to attention weights.
    """
    def __init__(self, embed_dim: int = 1536, seed: int = 0, topk: int = 8):
        self.embed_dim = int(embed_dim)
        self.rng = np.random.RandomState(int(seed))
        self.topk = int(topk)

    def _normalize_rows(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return (X / n).astype(np.float32)

    def propose(self, ids: List[str], vectors: np.ndarray) -> Tuple[List[PotentialFunction], np.ndarray, List[str]]:
        """
        Returns (potentials, attn_weights, labels) with len(labels)=len(attn_weights)=K.
        potentials[i].center is a raw encoder-space vector.
        """
        if len(ids) == 0 or vectors.size == 0:
            return [], np.zeros((0,), dtype=np.float32), []

        X = np.asarray(vectors, dtype=np.float32)
        if X.ndim != 2:
            X = X.reshape(len(ids), -1)

        # Novelty via distance to centroid
        mu = np.mean(X, axis=0, dtype=np.float32)
        dif = X - mu
        scores = np.linalg.norm(dif, axis=1)  # (N,)

        # Top-K
        k = max(1, min(self.topk, len(ids)))
        idx = np.argpartition(-scores, kth=k-1)[:k]
        # Sort the K for stable order
        idx = idx[np.argsort(-scores[idx])]

        labels = [ids[i] for i in idx]
        centers = X[idx]
        raw = scores[idx].astype(np.float32)

        # Normalize to soft weights
        w = raw - raw.max()
        w = np.exp(w)
        w = w / (w.sum() + 1e-12)

        potentials = [PotentialFunction(center=c, depth=float(wi), label=lab)
                      for c, wi, lab in zip(centers, w, labels)]
        return potentials, w.astype(np.float32), labels

# -----------------------------
# Hamiltonian mapping
# -----------------------------

def _to_root_potentials(mind, potentials: List[PotentialFunction]) -> np.ndarray:
    """
    Map Wavey potentials to a 240-dim node potential vector using the mind's
    TinyCompressor (mind.holo) and E8 physics root-nearest mapping.
    """
    V = np.zeros(240, dtype=np.float32)
    try:
        physics = getattr(mind, "physics", None)
        holo = getattr(mind, "holo", None)
        if physics is None or holo is None:
            return V
        for pot in potentials or []:
            center = getattr(pot, "center", None)
            depth  = float(getattr(pot, "depth", 0.0) or 0.0)
            if center is None or depth <= 0:
                continue
            c = np.asarray(center, dtype=np.float32)
            if c.size == 0:
                continue
            v8 = holo.encode(c)                        # map encoder-space → 8D
            idx = int(physics.find_nearest_root_index(v8))
            if 0 <= idx < 240:
                V[idx] += depth
    except Exception:
        pass
    return V

# -----------------------------
# Integration entrypoint
# -----------------------------

def integrate_one_cycle(mind, bridge: WaveyE8Bridge) -> Dict[str, Any]:
    """
    Pulls memory, proposes Wavey potentials/attention, nudges the mind (bias & attention),
    and returns a structured dict used by the main cognitive loop.
    Outputs include a 240-dim `node_potentials` aligned with the quantum engine.
    """
    # 1) Collect candidate memory items
    mem = getattr(mind, "memory", None)
    ids = []
    vecs = []
    try:
        if mem is not None and hasattr(mem, "main_vectors"):
            # Preserve stable order if available
            idx_order = getattr(mem, "_main_storage_ids", None)
            if isinstance(idx_order, (list, tuple)) and idx_order:
                for nid in idx_order:
                    v = mem.main_vectors.get(nid)
                    if isinstance(v, (list, tuple, np.ndarray)):
                        ids.append(nid); vecs.append(np.asarray(v, dtype=np.float32))
            else:
                # Fallback: iterate dict items
                for nid, v in mem.main_vectors.items():
                    ids.append(nid); vecs.append(np.asarray(v, dtype=np.float32))
    except Exception:
        pass

    if len(ids) == 0:
        # Nothing to do: return empty-safe outputs
        empty = np.zeros((0,), dtype=np.float32)
        return {
            "hamiltonian_bias": np.zeros(240, dtype=np.float32),
            "attention_weights": empty,
            "potentials": [],
            "events": [],
            "seed_used": False,
            "node_potentials": np.zeros(240, dtype=np.float32),
        }

    X = np.stack(vecs).astype(np.float32)  # (N,D)

    # 2) Propose Wavey candidates
    potentials, attn_weights, labels = bridge.propose(ids, X)

    # 3) Convert to 240-dim node potentials for the Hamiltonian
    node_potentials = _to_root_potentials(mind, potentials)

    # 4) Choose a Hamiltonian bias (use node_potentials directly; safe & sparse)
    bias = node_potentials

    # 5) Nudge the mind (optional hooks)
    try:
        if hasattr(mind, "apply_hamiltonian_bias"):
            mind.apply_hamiltonian_bias(np.asarray(bias, dtype=np.float32))
    except Exception:
        pass
    try:
        if hasattr(mind, "apply_attention_weights"):
            mind.apply_attention_weights(np.asarray(attn_weights, dtype=np.float32), labels=labels)
    except Exception:
        pass

    events = [{"type": "wavey.selection", "labels": labels, "k": len(labels)}]

    return {
        "hamiltonian_bias": np.asarray(bias, dtype=np.float32),
        "attention_weights": np.asarray(attn_weights, dtype=np.float32),
        "potentials": potentials,
        "events": events,
        "seed_used": True,
        "node_potentials": node_potentials.astype(np.float32),
    }
