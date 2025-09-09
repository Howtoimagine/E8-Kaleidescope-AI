"""
Memory structures extracted from the E8Mind monolith — full implementations.

This file implements:
 - NoveltyScorer
 - HopfieldModern
 - KanervaSDM
 - VSA (HRR via FFT circular convolution)
 - MicroReranker

Implementations follow the canonical behaviour described in the monolith and the
project's modular porting plan. They attempt to be robust in the absence of
external dependencies (LLM, trained VAE) and provide reasonable fallbacks.

Notes:
 - Some methods accept an optional `memory_manager` parameter to use manager
   helpers (find_similar_in_main_storage, get_average_nearest_neighbor_distance).
 - Async LLM coherence call is implemented as a safe stub returning 0.0 when no
   LLM client is available; this preserves the API while avoiding hard LLM deps.
"""

from __future__ import annotations
from typing import Any, Iterable, List, Tuple, Optional, Sequence
import numpy as np
import math
import asyncio

# Re-export EmergenceSeed to keep import surface stable
from core.data_structures import EmergenceSeed

# Try to import a projector from neural.autoencoder if available
try:
    from neural import autoencoder as _autoencoder_module  # type: ignore
except Exception:
    _autoencoder_module = None  # type: ignore


def _l2_normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    if n <= eps:
        return v
    return (v / (n + eps)).astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    s = float(np.sum(e)) + 1e-12
    return e / s


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class NoveltyScorer:
    """
    Novelty scoring:

      novelty = nearest_distance / average_nearest_neighbor_distance

    clipped to [0, 2].

    This class supports two usage patterns:

    1) calculate_novelty(vector, memory_manager=...) -> uses memory manager to
       find nearest neighbor and average neighborhood distance.

    2) score(vector, neighbor_vectors) -> uses the provided neighbor vectors
       (keeps pre-existing `score` compatibility).
    """

    def __init__(self, embed_dim: int = 1536, window: int = 256, seed: int = 42):
        self.embed_dim = int(embed_dim)
        self.window = int(window)
        self.seed = int(seed)

    def calculate_novelty(
        self,
        vector: np.ndarray,
        memory_manager: Optional[Any] = None,
        k: int = 1,
    ) -> float:
        """
        Compute novelty using the memory manager when available.

        memory_manager must expose:
         - find_similar_in_main_storage(query_vector, k) -> list of (id, vec)
         - get_average_nearest_neighbor_distance() -> float

        If memory_manager is not provided, falls back to returning 2.0 (max novelty)
        when neighbors aren't available.
        """
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        v = _l2_normalize_vec(v)

        neighbors = None
        avg_nn = None
        if memory_manager is not None:
            try:
                res = memory_manager.find_similar_in_main_storage(v, k=k)
                # Accept multiple result shapes: list of tuples (id, vec) or (vec,) or np.ndarray
                if res is None:
                    neighbors = []
                else:
                    # Normalize to sequence of vectors
                    if isinstance(res, np.ndarray):
                        neighbors = list(res)
                    else:
                        neighbors = []
                        for item in res:
                            if item is None:
                                continue
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                # (id, vec, ...)
                                vec = item[1]
                                if vec is None:
                                    continue
                                neighbors.append(np.asarray(vec, dtype=np.float32).reshape(-1))
                            elif isinstance(item, (list, tuple)) and len(item) == 1:
                                if item[0] is None:
                                    continue
                                neighbors.append(np.asarray(item[0], dtype=np.float32).reshape(-1))
                            elif isinstance(item, np.ndarray):
                                neighbors.append(item.reshape(-1))
                            else:
                                # Unknown format: skip
                                continue
                try:
                    avg_nn = float(memory_manager.get_average_nearest_neighbor_distance())
                except Exception:
                    avg_nn = None
            except Exception:
                neighbors = None
                avg_nn = None

        if not neighbors:
            # No neighbors -> maximum novelty
            return 2.0

        N = np.stack([_l2_normalize_vec(n) for n in neighbors], axis=0)  # (m, d)
        sims = (N @ v).squeeze(-1)
        dists = 1.0 - sims
        d_nearest = float(np.min(dists))

        # If avg_nn available use it; else compute local avg nearest neighbor distance within N
        if avg_nn is None:
            if N.shape[0] >= 2:
                simsNN = N @ N.T
                np.fill_diagonal(simsNN, -np.inf)
                nn_dists = 1.0 - np.max(simsNN, axis=1)
                avg_nn = float(np.mean(nn_dists))
            else:
                avg_nn = float(np.mean(dists))

        denom = max(avg_nn, 1e-6)
        score = d_nearest / denom
        return float(np.clip(score, 0.0, 2.0))

    async def calculate_coherence(self, text: str, llm_client: Optional[Any] = None) -> float:
        """
        OPTIONAL: Ask an LLM for a numeric coherence score.
        Expects the llm_client to have an async `generate_numeric` or `generate` that
        returns a numeric value when the appropriate prompt is given.

        Fallback: return 0.0 if llm_client not provided or not responding.
        """
        if llm_client is None:
            return 0.0

        # Defensive attempt: try a few common call signatures
        try:
            if hasattr(llm_client, "generate_numeric"):
                val = await llm_client.generate_numeric(text)
                return float(val)
            elif hasattr(llm_client, "generate"):
                # generate may return a dict with 'content' or similar
                out = await llm_client.generate(text)
                if isinstance(out, dict):
                    # try to extract a number
                    c = out.get("content") or out.get("text") or ""
                    try:
                        return float(c)
                    except Exception:
                        return 0.0
                elif isinstance(out, (str,)):
                    try:
                        return float(out.strip())
                    except Exception:
                        return 0.0
        except Exception:
            return 0.0

        return 0.0

    # Backwards-compatible alias for older code
    def score(self, vector: np.ndarray, neighbor_vectors: Sequence[np.ndarray]) -> float:
        # If no neighbors provided, return maximum novelty fallback
        if neighbor_vectors is None or len(neighbor_vectors) == 0:
            return 2.0
        return self._score_with_neighbors(vector, neighbor_vectors)

    def _score_with_neighbors(self, vector: np.ndarray, neighbor_vectors: Sequence[np.ndarray]) -> float:
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        v = _l2_normalize_vec(v)
        N = np.stack([_l2_normalize_vec(n) for n in neighbor_vectors], axis=0) if len(neighbor_vectors) else np.zeros((0, v.shape[0]), dtype=np.float32)
        if N.size == 0:
            return 2.0

        sims = (N @ v).squeeze(-1)
        dists = 1.0 - sims
        d_nearest = float(np.min(dists))

        if N.shape[0] >= 2:
            simsNN = N @ N.T
            np.fill_diagonal(simsNN, -np.inf)
            nn_dists = 1.0 - np.max(simsNN, axis=1)
            avg_nn = float(np.mean(nn_dists))
        else:
            avg_nn = float(np.mean(dists))
        denom = max(avg_nn, 1e-6)
        score = d_nearest / denom
        return float(np.clip(score, 0.0, 2.0))


class HopfieldModern:
    """
    Modern Hopfield-like cleanup.

    API:
      - update_prototypes(nodes_vectors, ratings, top_k): choose top_k nodes (by rating)
        and set self._P (d, K) prototypes matrix (columns are prototypes).
      - clean_up(vector, steps=3, tau=0.1) -> refined vector
      - clean: alias to clean_up for compatibility
    """

    def __init__(self, beta: float = 1.0, steps: int = 3):
        self.beta = float(beta)
        self.steps = int(steps)
        # prototypes matrix P shape (d, K). None until update_prototypes called.
        self._P: Optional[np.ndarray] = None

    def update_prototypes(self, node_vectors: Sequence[np.ndarray], ratings: Sequence[float], top_k: int = 32) -> None:
        """
        Build prototypes from the top-K highest-rated node vectors.

        node_vectors: sequence of vectors shape (n, d)
        ratings: sequence of floats length n
        """
        if not node_vectors:
            self._P = None
            return

        vals = np.asarray(node_vectors, dtype=np.float32)
        ratings_arr = np.asarray(ratings, dtype=np.float32)
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
        n, d = vals.shape
        if ratings_arr.shape[0] != n:
            # resize or pad/truncate ratings to match node count
            ratings_arr = np.resize(ratings_arr, (n,))

        top_k = max(1, min(int(top_k), n))
        order = np.argsort(-ratings_arr)[:top_k]
        selected = vals[order]  # (K, d)
        # Normalize selected and place as columns in P (d, K)
        selected_n = np.stack([_l2_normalize_vec(v) for v in selected], axis=0).T
        # selected_n is (d, K)
        self._P = np.asarray(selected_n, dtype=np.float32)

    def clean_up(self, vector: np.ndarray, steps: Optional[int] = None, tau: float = 0.1) -> np.ndarray:
        """
        Clean a single vector using stored prototypes P.

        v <- P @ softmax( (P^T v) / tau )
        Iterate `steps` times (default self.steps). L2-normalize each iteration.
        Early-stop when delta < 1e-4.

        Returns cleaned vector of shape (d,).
        """
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        if self._P is None:
            # No prototypes: return normalized input
            return _l2_normalize_vec(v)

        P = self._P  # (d, K)
        if P.ndim != 2:
            return _l2_normalize_vec(v)
        d, K = P.shape
        if v.shape[0] != d:
            # dimension mismatch: try to reshape or project (best-effort: trim or pad)
            if v.shape[0] > d:
                v = v[:d]
            else:
                v = np.pad(v, (0, d - v.shape[0]))
        v_t = _l2_normalize_vec(v)
        steps = int(steps) if steps is not None else int(self.steps)
        tau = float(max(tau, 1e-6))

        PT = P.T  # (K, d)
        for _ in range(max(1, steps)):
            logits = (PT @ v_t) / tau  # (K,)
            weights = _softmax(logits)
            v_next = (P @ weights).reshape(-1)
            v_next = _l2_normalize_vec(v_next)
            if np.linalg.norm(v_next - v_t) < 1e-4:
                v_t = v_next
                break
            v_t = v_next
        return v_t.astype(np.float32)

    # Backwards compatibility
    def clean(self, vector: np.ndarray, steps: Optional[int] = None, tau: float = 0.1) -> np.ndarray:
        return self.clean_up(vector, steps=steps, tau=tau)


class KanervaSDM:
    """
    Kanerva Sparse Distributed Memory with 8D address projection.

    Behaviour:
     - If neural.autoencoder provides a projector, use it to get an 8D projection.
     - Addresses live in an 8D unit sphere. Writes update nearest addresses using
       a small learning rate and sign of projected vector (stable updates).
     - read_strength returns sigmoid((sum_hits - 10)/5.0)
    """

    def __init__(self, address_count: int = 2048, dim: int = 1536, seed: int = 42, radius: float = 0.85):
        self.address_count = int(address_count)
        self.dim = int(dim)
        self.seed = int(seed)
        self.radius = float(radius)
        self._initialized = False

        # fallback random projection if no autoencoder projector exists
        self._fallback_R: Optional[np.ndarray] = None

    def _ensure_init(self) -> None:
        if getattr(self, "_initialized", False):
            return
        rng = np.random.default_rng(self.seed)
        self._addr_dim = 8
        # fallback random projection matrix (8 x dim)
        self._fallback_R = rng.standard_normal((self._addr_dim, self.dim)).astype(np.float32)
        # addresses randomly initialized in 8D and normalized
        self._addresses = rng.standard_normal((self.address_count, self._addr_dim)).astype(np.float32)
        self._addresses /= (np.linalg.norm(self._addresses, axis=1, keepdims=True) + 1e-12)
        # slot values in original dim
        self._values = np.zeros((self.address_count, self.dim), dtype=np.float32)
        # per-address hit counters
        self._hits = np.zeros((self.address_count,), dtype=np.int32)
        self._initialized = True

    def _get_vec8d(self, vec: np.ndarray) -> Optional[np.ndarray]:
        """
        Project vec to 8D using autoencoder projector if available, else use fallback.
        Returns unit-normalized 8D vector.
        """
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        # Try external autoencoder module
        try:
            # Prefer deterministic projector from neural.autoencoder if available
            if _autoencoder_module is not None and hasattr(_autoencoder_module, "SubspaceProjector"):
                Projector = getattr(_autoencoder_module, "SubspaceProjector")
                proj = Projector(seed=self.seed)
                res = proj.project_to_dim(v, target_dim=8, normalize=True)
                return _l2_normalize_vec(np.asarray(res, dtype=np.float32))
        except Exception:
            # fallback to local projection
            pass

        # Fallback random projection
        self._ensure_init()
        R = self._fallback_R
        if R is None:
            # Defensive: ensure initialized; if still None, return zero vector
            self._ensure_init()
            R = self._fallback_R
            if R is None:
                return _l2_normalize_vec(np.zeros((8,), dtype=np.float32))
        a = (R @ v).astype(np.float32)
        return _l2_normalize_vec(a)

    def write(self, key: np.ndarray, value: np.ndarray) -> None:
        """
        Write value into SDM using projected key in 8D.
        Updates nearest addresses (k up to 32) with learning rate 0.05 and
        make small sign-based updates if only sign info is available.
        """
        self._ensure_init()
        a = self._get_vec8d(key)
        if a is None:
            return

        # similarities to addresses (cosine since unit-norm)
        sims = (self._addresses @ a).astype(np.float32)
        # select top-k neighbors (bounded)
        k = min(32, self.address_count)
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        weights = sims[idx]
        weights = np.clip(weights, 0.0, None)
        s = float(weights.sum())
        if s > 0:
            weights = weights / (s + 1e-12)
        else:
            weights = np.ones_like(weights) / float(len(weights))

        value = np.asarray(value, dtype=np.float32).reshape(-1)
        lr = 0.05
        for ii, w in zip(idx, weights):
            # update values: blend old with new scaled by similarity weight
            self._values[ii] = (1.0 - lr) * self._values[ii] + lr * (w * value)
            self._hits[ii] += 1

    def read(self, key: np.ndarray, k: int = 8) -> np.ndarray:
        """
        Read a value near a key address using k nearest addresses.
        Aggregates address slot values weighted by similarity; returns a normalized vector.
        """
        self._ensure_init()
        a = self._get_vec8d(key)
        if a is None:
            return np.zeros((self.dim,), dtype=np.float32)

        sims = (self._addresses @ a).astype(np.float32)
        kk = min(max(1, int(k)), sims.shape[0])
        idx = np.argpartition(-sims, kth=kk - 1)[:kk]
        w = sims[idx]
        w = np.clip(w, 0.0, None)
        s = float(w.sum())
        if s > 0:
            w = w / (s + 1e-12)
        else:
            w = np.ones_like(w) / len(w)

        out = np.zeros((self.dim,), dtype=np.float32)
        for ii, weight in zip(idx, w):
            out += weight * self._values[ii]

        nrm = np.linalg.norm(out)
        if nrm > 0:
            out = out / nrm
        return out.astype(np.float32)

    def read_strength(self, key: np.ndarray) -> float:
        """
        Return a scalar indicating how strongly this key maps to memory:
          sigmoid((sum_hits - 10) / 5.0)

        If no addresses exceed radius threshold, fallback to sigmoid((mean_hits -10)/5.0).
        """
        self._ensure_init()
        a = self._get_vec8d(key)
        if a is None:
            return 0.0
        sims = (self._addresses @ a).astype(np.float32)
        # addresses within radius
        mask = sims >= self.radius
        sum_hits = int(self._hits[mask].sum()) if mask.any() else int(self._hits.sum())
        val = _sigmoid((float(sum_hits) - 10.0) / 5.0)
        return float(val)


class VSA:
    """
    HRR-style Vector Symbolic Architecture utilities using FFT circular convolution.
    Includes role vectors (PARENT_A, PARENT_B, CAUSE, EFFECT) and parentage encoding.
    """

    def __init__(self, dim: int = 1536, seed: int = 42):
        self.dim = int(dim)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        # role vectors
        self.PARENT_A = _l2_normalize_vec(rng.standard_normal((self.dim,)).astype(np.float32))
        self.PARENT_B = _l2_normalize_vec(rng.standard_normal((self.dim,)).astype(np.float32))
        self.CAUSE = _l2_normalize_vec(rng.standard_normal((self.dim,)).astype(np.float32))
        self.EFFECT = _l2_normalize_vec(rng.standard_normal((self.dim,)).astype(np.float32))

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        if a.shape != b.shape:
            raise ValueError("bind expects vectors of same shape")
        fa = np.fft.rfft(a)
        fb = np.fft.rfft(b)
        out = np.fft.irfft(fa * fb, n=a.shape[0])
        return _l2_normalize_vec(out)

    def unbind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        if a.shape != b.shape:
            raise ValueError("unbind expects vectors of same shape")
        fa = np.fft.rfft(a)
        fb = np.fft.rfft(b)
        out = np.fft.irfft(fa * np.conj(fb), n=a.shape[0])
        return _l2_normalize_vec(out)

    def superpose(self, vectors: Iterable[np.ndarray]) -> np.ndarray:
        vecs = [np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors if v is not None]
        if len(vecs) == 0:
            return np.zeros((self.dim,), dtype=np.float32)
        out = np.sum(vecs, axis=0)
        return _l2_normalize_vec(out)

    def encode_parentage(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        encode_parentage(a, b) = normalize(bind(PARENT_A, a) + bind(PARENT_B, b))
        """
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        part_a = self.bind(self.PARENT_A, a)
        part_b = self.bind(self.PARENT_B, b)
        return _l2_normalize_vec(part_a + part_b)


class MicroReranker:
    """
    Lightweight reranker used for candidate validation.

    Features: [coherence, novelty, ppl, dup_rate, parent_coh_mean]
    Weights:  [  0.4   ,  0.3  , -0.1,  -0.2   ,   0.15       ]
    """

    def __init__(self, top_k: int = 8):
        self.top_k = int(top_k)
        self.weights = np.asarray([0.4, 0.3, -0.1, -0.2, 0.15], dtype=np.float32)

    def rerank(self, query: np.ndarray, candidates: List[Tuple[str, np.ndarray]]) -> List[Tuple[str, float]]:
        """
        Existing rerank implementation preserved for backwards compatibility.
        """
        if not candidates:
            return []

        q = _l2_normalize_vec(query)
        ids: List[str] = []
        vecs: List[np.ndarray] = []
        for cid, v in candidates:
            vv = _l2_normalize_vec(v)
            ids.append(cid)
            vecs.append(vv)

        V = np.stack(vecs, axis=0)
        sims = V @ q
        if V.shape[0] > 1:
            cross = V @ V.T
            np.fill_diagonal(cross, 0.0)
            avg_to_others = cross.sum(axis=1) / (V.shape[0] - 1)
        else:
            avg_to_others = np.zeros((V.shape[0],), dtype=np.float32)

        scores = sims - 0.25 * avg_to_others
        order = np.argsort(-scores)
        top = order[: max(1, int(self.top_k))]
        return [(ids[i], float(scores[i])) for i in top]

    def _compute_candidate_score(self, coherence: float, novelty: float, ppl: float, dup_rate: float, parent_coh_mean: float) -> float:
        feats = np.asarray([coherence, novelty, ppl, dup_rate, parent_coh_mean], dtype=np.float32)
        return float(np.dot(self.weights, feats))

    def validate(
        self,
        candidate_vec: np.ndarray,
        parent_ids: Optional[Sequence[str]] = None,
        novelty: Optional[float] = None,
        coherence: Optional[float] = None,
        memory_manager: Optional[Any] = None,
        margin: float = 0.1,
    ) -> bool:
        """
        Validate a candidate against hard negatives.

        Procedure:
         - Build candidate score from provided novelty/coherence and fallback metrics.
         - Retrieve neighbors via memory_manager.find_similar_in_main_storage if provided.
         - Collect "hard negatives": neighbors with rating < 0.45 (if available) or low similarity.
         - Reject candidate if candidate_score <= (max_negative_score + margin)
        """
        cand = _l2_normalize_vec(candidate_vec)
        # Fallback feature values
        coh = float(coherence) if coherence is not None else 0.0
        nov = float(novelty) if novelty is not None else 1.0
        ppl = 0.0
        dup_rate = 0.0
        parent_coh_mean = 0.0

        # Attempt to get more accurate parent_coh_mean via memory_manager if available
        if memory_manager is not None and parent_ids:
            try:
                parent_vecs = []
                for pid in parent_ids:
                    pv = memory_manager.get_vector_by_id(pid)
                    if pv is not None:
                        parent_vecs.append(_l2_normalize_vec(pv))
                if parent_vecs:
                    # coherence between candidate and parents (mean cosine)
                    sims = [float(np.dot(cand, pv)) for pv in parent_vecs]
                    parent_coh_mean = float(np.mean(sims))
            except Exception:
                parent_coh_mean = parent_coh_mean

        candidate_score = self._compute_candidate_score(coh, nov, ppl, dup_rate, parent_coh_mean)

        # Gather hard negatives
        neg_scores: List[float] = []
        if memory_manager is not None:
            try:
                neighbors = memory_manager.find_similar_in_main_storage(cand, k=self.top_k)
                # neighbors may be sequence of tuples or raw vectors
                for nb in neighbors or []:
                    nb_vec = None
                    nb_rating = None
                    if isinstance(nb, (list, tuple)):
                        # try to infer vector and metadata
                        if len(nb) >= 2:
                            raw_vec = nb[1]
                            if raw_vec is None:
                                continue
                            nb_vec = np.asarray(raw_vec, dtype=np.float32)
                        if len(nb) >= 3:
                            # third element may be metadata/dict containing rating
                            meta = nb[2]
                            if isinstance(meta, dict) and "rating" in meta:
                                val = meta.get("rating", None)
                                if isinstance(val, (int, float, np.floating, str)):
                                    try:
                                        nb_rating = float(val)
                                    except Exception:
                                        nb_rating = None
                                else:
                                    nb_rating = None
                    elif isinstance(nb, np.ndarray):
                        nb_vec = nb
                    if nb_vec is None:
                        continue
                    nb_vec = _l2_normalize_vec(nb_vec)
                    # best-effort: obtain neighbor features. If rating provided, use it, else compute simple coherence
                    if nb_rating is not None:
                        # if rating is already an overall score, convert to a comparable feature set
                        neg_score_val = float(nb_rating)
                    else:
                        # fallback: compute coherence-like score between candidate and neighbor
                        neg_score_val = float(np.dot(cand, nb_vec))
                    neg_scores.append(neg_score_val)
            except Exception:
                neg_scores = []

        if not neg_scores:
            # No hard negatives: accept if candidate_score > margin threshold relative to zero
            return candidate_score > (0.0 + margin)

        max_neg = float(np.max(neg_scores))
        # If candidate_score is not greater than max_neg + margin, reject
        return candidate_score > (max_neg + float(margin))


__all__ = [
    "NoveltyScorer",
    "HopfieldModern",
    "KanervaSDM",
    "VSA",
    "MicroReranker",
    "EmergenceSeed",
]
