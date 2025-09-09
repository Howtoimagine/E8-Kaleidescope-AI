from __future__ import annotations
import time
from typing import Tuple
import numpy as np

try:
    import faiss  # type: ignore
    _FAISS = True
except Exception:
    _FAISS = False

try:
    from sklearn.neighbors import KDTree as _SKKDTree  # type: ignore
except Exception:
    _SKKDTree = None

try:
    from sklearn.metrics.pairwise import cosine_distances as _sk_cosine_distances, cosine_similarity as _sk_cosine_similarity  # type: ignore
except Exception:
    _sk_cosine_distances, _sk_cosine_similarity = None, None

try:
    from scipy.spatial import KDTree as _SPKDTree  # type: ignore
except Exception:
    _SPKDTree = None


class KDTree:
    """Unified KDTree wrapper with FAISS/scikit-learn/scipy/NumPy backends."""
    def __init__(self, data):
        X = np.asarray(data, dtype=np.float32)
        if X.ndim != 2:
            X = np.atleast_2d(X)
        self.n = int(X.shape[0])
        self._latency_ms = []

        if _FAISS and self.n > 0:
            self._is_faiss = True
            self._is_fallback = False
            dim = int(X.shape[1])
            self._faiss_index = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            Xn = X / norms
            self._faiss_index.add(Xn)  # type: ignore[attr-defined]
            self._impl = None
        elif _SKKDTree is not None:
            self._impl = _SKKDTree(X)  # type: ignore[call-arg]
            self._is_faiss = False
            self._is_fallback = False
        elif _SPKDTree is not None:
            self._impl = _SPKDTree(X)  # type: ignore[call-arg]
            self._is_faiss = False
            self._is_fallback = False
        else:
            self._impl = X
            self._is_faiss = False
            self._is_fallback = True

    def query(self, q, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        q_arr = np.asarray(q, dtype=np.float32)
        is_single = (q_arr.ndim == 1)
        Q = np.atleast_2d(q_arr)

        if getattr(self, "_is_faiss", False):
            norms = np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
            Qn = (Q / norms).astype(np.float32)
            if Qn.ndim == 1:
                Qn = Qn.reshape(1, -1)
            try:
                D, I = self._faiss_index.search(Qn, int(k))  # type: ignore[attr-defined]
            except Exception:
                D = np.ones((Q.shape[0], int(k)), dtype=np.float32)
                I = -np.ones((Q.shape[0], int(k)), dtype=np.int64)
            d = 1.0 - D
            i = I.astype(np.int64, copy=False)
        elif (self._impl is not None) and hasattr(self._impl, "query"):
            d, i = self._impl.query(Q, k=int(k))  # type: ignore[attr-defined]
            d = np.asarray(d, dtype=np.float32, copy=False)
            i = np.asarray(i, dtype=np.int64, copy=False)
        else:
            data = np.asarray(self._impl, dtype=np.float32)
            all_dists = []
            all_idxs = []
            for qv in Q:
                dv = np.sqrt(np.sum((data - qv) ** 2, axis=1))
                kk = int(k)
                if kk < self.n:
                    part = np.argpartition(dv, kk - 1)[:kk]
                    local = np.argsort(dv[part])
                    idx = part[local]
                else:
                    idx = np.argsort(dv)[:kk]
                all_idxs.append(idx.astype(np.int64))
                all_dists.append(dv[idx].astype(np.float32))
            d = np.stack(all_dists, axis=0) if len(all_dists) > 1 else np.asarray(all_dists[0]).reshape(1, -1)
            i = np.stack(all_idxs, axis=0) if len(all_idxs) > 1 else np.asarray(all_idxs[0]).reshape(1, -1)

        d_out = d.ravel() if is_single else d
        i_out = i.ravel() if is_single else i

        try:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._latency_ms.append(dt_ms)
            if len(self._latency_ms) > 128:
                self._latency_ms.pop(0)
        except Exception:
            pass
        return d_out, i_out


def cosine_distances(A, B):
    if _sk_cosine_distances is not None:
        return _sk_cosine_distances(A, B)
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    def _norm(x): return np.sqrt((x * x).sum(axis=1, keepdims=True)) + 1e-12
    A2 = A / _norm(A)
    B2 = B / _norm(B)
    return 1.0 - (A2 @ B2.T)


def cosine_similarity(A, B):
    if _sk_cosine_similarity is not None:
        return _sk_cosine_similarity(A, B)
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    def _norm(x): return np.sqrt((x * x).sum(axis=1, keepdims=True)) + 1e-12
    A2 = A / _norm(A)
    B2 = B / _norm(B)
    return A2 @ B2.T


__all__ = ["KDTree", "cosine_distances", "cosine_similarity"]
