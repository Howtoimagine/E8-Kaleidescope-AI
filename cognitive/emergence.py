from __future__ import annotations

from typing import List, Tuple
import numpy as np

def detect_emergence_minima(V: np.ndarray, top_k: int = 3, sigma_threshold: float = 2.0) -> List[Tuple[int, float]]:
    """
    Detect salient minima in a potential field V (lower is better).
    - Computes z-scores over V and keeps indices with value <= mean - sigma_threshold * std.
    - Returns up to top_k lowest (index, value) pairs sorted by value asc.
    """
    if V is None or not isinstance(V, np.ndarray) or V.size == 0:
        return []

    v = V.astype(np.float32).reshape(-1)
    mu = float(np.mean(v))
    sd = float(np.std(v)) + 1e-12

    # Threshold mask for salient minima
    thresh = mu - sigma_threshold * sd
    mask = v <= thresh

    candidates = np.where(mask)[0]
    if candidates.size == 0:
        # Fallback: take absolute minima
        k = max(1, min(int(top_k), v.shape[0]))
        idx = np.argpartition(v, k - 1)[:k]
        idx = idx[np.argsort(v[idx])]
        return [(int(i), float(v[i])) for i in idx]

    # Rank masked candidates by v value asc
    vals = v[candidates]
    order = np.argsort(vals)
    chosen = candidates[order][: max(1, int(top_k))]
    return [(int(i), float(v[i])) for i in chosen]
