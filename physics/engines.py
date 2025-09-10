from __future__ import annotations
from typing import Any, List, Optional, Sequence, Tuple, Dict
import numpy as np

from .e8_lattice import E8Physics, weyl_average_potential, add_curiosity_penalty
from .potential_fields import MultiAnchorField

# Optional scipy imports for quantum engine
try:
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import expm_multiply, eigsh
except ImportError:
    csr_matrix = None
    diags = None
    expm_multiply = None
    eigsh = None

# Optional sklearn for PCA
try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

def build_adjacency_240_from_memory(memory, mind, alpha=1.0, decay_tau=600.0, reward_gain=0.5):
    """Build a 240x240 adjacency matrix from memory connections."""
    try:
        # Get memory connections
        vectors, ids = memory.get_all_vectors()
        if vectors is None or len(vectors) == 0:
            # Return default lattice connections
            return mind.physics.weights
        
        # Use basic approach based on embedding similarities
        n = len(vectors)
        sim_matrix = np.dot(vectors, vectors.T)
        
        # Map to 240x240 based on hash of memory IDs
        adj_240 = np.zeros((240, 240), dtype=np.float32)
        for i in range(min(n, 240)):
            for j in range(min(n, 240)):
                if i != j:
                    adj_240[i, j] = max(0, sim_matrix[i, j] * alpha)
        
        return adj_240
    except Exception:
        # Fallback to physics weights
        return getattr(mind, 'physics', {}).get('weights', np.eye(240, dtype=np.float32))

def normalize_vector(v):
    """Normalize a vector to unit length."""
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm

class TinyCompressor:
    """Simple autoencoder-like compressor."""
    def __init__(self, in_dim: int, code_dim: int = 8):
        self.in_dim = in_dim
        self.code_dim = code_dim
        # Simple linear projection
        self.W = np.random.randn(in_dim, code_dim).astype(np.float32) * 0.1
    
    def encode(self, x):
        """Encode input to lower dimension."""
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if len(x) != self.in_dim:
            # Pad or truncate
            if len(x) < self.in_dim:
                x = np.pad(x, (0, self.in_dim - len(x)))
            else:
                x = x[:self.in_dim]
        return x @ self.W

class QuantumEngine:
    def attach_mind(self, mind_instance):
        self.mind = mind_instance
        return self
    
    def __init__(self, physics, config, console: Any):
        self.mind = None
        self._last_graph_build_step = -1

        self.console = console
        self.physics, self.config = physics, config
        self.psi = np.ones((config.batch, 240), dtype=np.complex64) / np.sqrt(240)
        self.rng = np.random.default_rng(config.seed)
        self.H: Any = None
        self._last_H: Any = None
        self._last_potential: Optional[np.ndarray] = None
        self._last_norm = np.nan
        self._last_energy = np.nan
        self.build_hamiltonian()
        self.console.log("[INIT] Quantum Engine online (Non-Linear Edition).")

    def build_hamiltonian(self, V: Optional[np.ndarray] = None):
        if diags is None or csr_matrix is None:
            # Fallback: store dense H using numpy
            if V is None:
                V = np.zeros(240, dtype=np.float32)
            H = (self.config.gamma * np.asarray(self.physics.L_norm, dtype=np.complex64)) + np.diag(V)
            self.H = H
            self._last_H = H
            self._last_potential = np.asarray(V).copy()
            return
        if V is None:
            V = np.zeros(240, dtype=np.float32)
        # Safer guard for optional mind/memory
        _mind = getattr(self, 'mind', None)
        _mem = getattr(_mind, 'memory', None) if _mind is not None else None
        use_graph = (getattr(self.config, 'mode', 'lattice') == 'graph' and _mind is not None and _mem is not None)
        if use_graph and (self._last_graph_build_step < 0 or (getattr(_mind, 'step_num', 0) - self._last_graph_build_step) >= getattr(self.config,'rebuild_every',10)):
            A = build_adjacency_240_from_memory(_mem, _mind,
                                               alpha=getattr(self.config,'alpha_edge',1.0),
                                               decay_tau=getattr(self.config,'decay_tau',600.0),
                                               reward_gain=getattr(self.config,'reward_gain',0.5))
            if csr_matrix is None or diags is None:
                H = (-self.config.gamma * np.asarray(A, dtype=np.complex64)) + getattr(self.config,'lambda_V',0.2) * np.diag(V)
                self.H = H
            else:
                H = (-self.config.gamma * A.astype(np.complex64)) + getattr(self.config,'lambda_V',0.2) * diags(V)
                self.H = csr_matrix(H)
            self._last_graph_build_step = getattr(_mind, 'step_num', 0)
        else:
            if diags is None or csr_matrix is None:
                H = (self.config.gamma * np.asarray(self.physics.L_norm, dtype=np.complex64)) + np.diag(V)
                self.H = H
            else:
                H = (self.config.gamma * self.physics.L_norm.astype(np.complex64)) + diags(V)
                self.H = csr_matrix(H)
        self._last_H = self.H
        self._last_potential = np.asarray(V).copy()

    def _probs(self):
        p = np.abs(self.psi)**2
        return p / (np.sum(p, axis=1, keepdims=True) + 1e-12)

    def step_adaptive(self, tv_target=0.07, dt_min=0.02, dt_max=1.2):
        if self.H is None:
            return 0.0

        p0 = self._probs()
        H_eff = self.H.copy()
        if self.config.non_linearity_strength != 0:
            feedback = self.config.non_linearity_strength * p0[0]
            if diags is not None and hasattr(H_eff, 'shape') and csr_matrix is not None:
                try:
                    H_eff = H_eff + diags(feedback.astype(np.float32), 0)
                except Exception:
                    pass
            else:
                # dense numpy path
                H_eff = np.asarray(H_eff) + np.diag(feedback.astype(np.float32))

        if expm_multiply is not None and hasattr(H_eff, 'dot'):
            psi_new = expm_multiply(-1j * H_eff * self.config.dt, self.psi.T).T
        else:
            # crude Euler step fallback for environments without scipy
            H_dense = np.asarray(H_eff)
            psi_new = (self.psi + (-1j * self.config.dt) * (self.psi @ H_dense.T)).astype(np.complex64)
        nrm = np.linalg.norm(psi_new, axis=1, keepdims=True)
        self.psi = psi_new / np.maximum(nrm, 1e-12)
        p1 = self._probs()

        tv = 0.5 * float(np.abs(p0 - p1).sum(axis=1).mean())
        if tv < 0.5*tv_target: self.config.dt = min(dt_max, self.config.dt*1.25)
        elif tv > 1.5*tv_target: self.config.dt = max(dt_min, self.config.dt*0.66)

        if self.config.dephase > 0:
            mag = np.abs(self.psi)
            self.psi = (1.0 - self.config.dephase) * self.psi + self.config.dephase * mag
            nrm = np.linalg.norm(self.psi, axis=1, keepdims=True)
            self.psi /= np.maximum(nrm, 1e-12)

        try:
            self._last_norm = float(np.mean(np.sum(np.abs(self.psi)**2, axis=1)))
            Href = self._last_H
            if Href is not None and getattr(Href, 'ndim', 0) == 2:
                Energies = []
                for b in range(self.psi.shape[0]):
                    v = self.psi[b].reshape(-1,1)
                    E = (np.conjugate(v).T @ (Href @ v)).ravel()[0]
                    Energies.append(np.real(E))
                self._last_energy = float(np.mean(Energies))
        except Exception:
            self._last_norm = np.nan
            self._last_energy = np.nan
        return tv

    def measure_local(self, prev_idx, sigma=None):
        sigma = sigma or self.config.locality_sigma
        P = self._probs()
        masks = np.stack([self.physics.heat_mask_cached(i, sigma) for i in prev_idx]) if isinstance(prev_idx, (list, np.ndarray)) else np.tile(self.physics.heat_mask_cached(int(prev_idx), sigma), (self.config.batch, 1))
        P *= masks
        P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
        return np.array([self.rng.choice(P.shape[1], p=p) for p in P], dtype=np.int32)

    def measure_hybrid(self, prev_idx=None, sigma=None, topk=None):
        """Hybrid measurement: combine engine amplitudes with a soft projection mask
        derived from the last potential (attractive wells), then apply local heat-mask
        around the previous index. Falls back to measure_local if data is missing.
        Returns a list of chosen indices (len=batch).
        """
        if prev_idx is None:
            prev_idx = 0

        if not hasattr(self, "psi"):
            return self.measure_local([prev_idx] * self.config.batch, sigma)
        B, N = self.psi.shape

        P = np.abs(self.psi)**2
        P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)

        Vlast = self._last_potential
        if Vlast is not None and np.size(Vlast) == N:

            soft = np.maximum(0.0, -np.real(np.asarray(Vlast).reshape(1, -1)))
            if topk is None:
                topk = int(getattr(self.config, "topk_amp", 5) or 5)

            idx = np.argpartition(soft[0], -topk)[-topk:]
            mask = np.zeros_like(P)
            mask[:, idx] = 1.0

            Amp = np.sqrt(P) * np.sqrt(soft + 1e-12)
            P = (Amp**2) * mask
            P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)

        if sigma is None:
            sigma = float(getattr(self.config, "locality_sigma", 1.5) or 1.5)

        hops = self.physics.hops
        w = np.exp(-(hops[prev_idx]**2) / (2.0 * sigma * sigma))

        P = P * w.reshape(1, -1)
        P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)

        choices = []
        for b in range(B):
            choices.append(int(self.rng.choice(N, p=P[b])))
        return choices

    def telemetry_state(self):
        """Return latest quantum telemetry values."""
        return {
            "dt": float(getattr(self.config, "dt", 0.01)),
            "gamma": float(getattr(self.config, "gamma", 1.0)),
            "dephase": float(getattr(self.config, "dephase", 0.0)),
            "norm": float(self._last_norm),
            "energy": float(self._last_energy),
            "topk_amp": int(getattr(self.config, "topk_amp", 5)),
            "locality_sigma": float(getattr(self.config, "locality_sigma", 1.5)),
        }

    def measure_ablation(self, prev_idx:int, sigma: Optional[float]=None, window:int=5, trials:int=512):
        """Compare local vs hybrid measurement near prev_idx.
        Returns dict with hit counts and rates inside ±window."""
        if sigma is None:
            sigma = float(getattr(self.config, "locality_sigma", 1.5) or 1.5)
        B, N = getattr(self, "psi", np.zeros((1,1))).shape
        total = trials * B
        if total == 0:
            return {}

        local_choices = []
        hybrid_choices = []
        for _ in range(trials):
            local_choices.extend(self.measure_local([prev_idx] * B, sigma=sigma))
            hybrid_choices.extend(self.measure_hybrid(prev_idx=prev_idx, sigma=sigma))
        local_counts = np.bincount(np.asarray(local_choices), minlength=N)
        hybrid_counts = np.bincount(np.asarray(hybrid_choices), minlength=N)

        lo = max(0, prev_idx-window); hi = min(N-1, prev_idx+window)
        local_win = int(local_counts[lo:hi+1].sum())
        hybrid_win = int(hybrid_counts[lo:hi+1].sum())
        return {
            "prev_idx": int(prev_idx),
            "window": int(window),
            "sigma": float(sigma),
            "trials": int(trials),
            "batch": int(B),
            "N": int(N),
            "local_win": local_win,
            "hybrid_win": hybrid_win,
            "local_rate": float(local_win/total),
            "hybrid_rate": float(hybrid_win/total),
        }

class ClassicalEngine:
    def __init__(self, physics, config, console: Any):
        self.console = console
        self.physics, self.config = physics, config
        self.rng = np.random.default_rng(config.seed)
        self.console.log("[INIT] Classical Engine online.")

    def next_index(self, prev_idx, sensor8):
        nbrs = np.where(self.physics.weights[prev_idx] > 0)[0]
        if nbrs.size > 0:
            if np.linalg.norm(sensor8) > 0:
                scores = self.physics.roots[nbrs] @ sensor8
                p = np.exp(2.5 * scores); p /= np.sum(p)
                return self.rng.choice(nbrs, p=p)
            return self.rng.choice(nbrs)
        return self.rng.integers(0, 240)

class E8BoundaryFabric:
    def __init__(self, physics: "E8Physics", seed: int = 1337):
        self.physics = physics
        self.N = physics.roots.shape[0]
        self.A = (physics.weights > 0).astype(np.float32)
        self.pos2d: Optional[np.ndarray] = None
        self.z1d: Optional[np.ndarray] = None
        self.rng = np.random.default_rng(seed)

    def layout_2d(self):
        W = self.A; deg = W.sum(axis=1)
        Dm12 = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
        L = np.eye(self.N, dtype=np.float32) - (Dm12[:,None] * W * Dm12[None,:])
        try:
            if csr_matrix is None or eigsh is None: raise RuntimeError("scipy is required for layout_2d")
            _, vecs = eigsh(csr_matrix(L), k=4, which='SM')
            P = vecs[:, 1:4]
        except Exception:
            _, vecs = np.linalg.eigh(L)
            P = vecs[:, 1:4]
        P = (P - P.mean(axis=0)) / (P.std(axis=0) + 1e-6)
        self.pos2d = P[:, :2].astype(np.float32)
        self.z1d = P[:, 2].astype(np.float32)

    def neighbors(self, i: int) -> np.ndarray:
        return np.where(self.A[i] > 0)[0].astype(np.int32)

    def to_json(self):
        if self.pos2d is None: self.layout_2d()
        edges = np.column_stack(np.where(np.triu(self.A, 1) > 0)).tolist()
        if self.pos2d is None or self.z1d is None:
            return {"nodes": [], "edges": []}
        return {
            "nodes": [{"id": int(i), "x": float(self.pos2d[i,0]), "y": float(self.pos2d[i,1]), "z": float(self.z1d[i])} for i in range(self.N)],
            "edges": [{"s": int(i), "t": int(j)} for i, j in edges]
        }

class SliceStack:
    def __init__(self, n_slices: int = 24, zmin: float = -1.5, zmax: float = 1.5):
        self.n, self.zmin, self.zmax = n_slices, zmin, zmax
        self.bin = np.linspace(self.zmin, self.zmax, self.n + 1)

    def index(self, z: float) -> int:
        return int(np.clip(np.searchsorted(self.bin, z, side="right") - 1, 0, self.n - 1))

class HoloEncoder:
    def __init__(self, fabric: E8BoundaryFabric, feat_dim: int = 8, shadow_k: int = 12, seed: int = 1337):
        self.fabric, self.feat_dim, self.shadow_k = fabric, feat_dim, shadow_k
        self.rng = np.random.default_rng(seed)
        self._U_cache: Dict[Tuple, np.ndarray] = {}
        self.store: Dict[Tuple[int, int], float] = {}

    def shadow_set(self, bulk_idx: int, pos_hint_xy: Optional[np.ndarray] = None) -> np.ndarray:
        if pos_hint_xy is not None and self.fabric.pos2d is not None:
            d = np.sum((self.fabric.pos2d - pos_hint_xy[None,:])**2, axis=1)
            return np.argsort(d)[:self.shadow_k].astype(np.int32)
        nb = self.fabric.neighbors(int(bulk_idx))
        if nb.size >= self.shadow_k: return nb[:self.shadow_k]
        pool = np.setdiff1d(np.arange(self.fabric.N), np.append(nb, bulk_idx))
        if not pool.size > 0: return nb
        extra_count = self.shadow_k - nb.size
        extra = self.rng.choice(pool, size=min(extra_count, pool.size), replace=False)
        return np.concatenate([nb, extra]).astype(np.int32)

    def _U(self, shadow_ids: np.ndarray):
        key = tuple(sorted(shadow_ids.tolist()))
        if key not in self._U_cache:
            K, D = len(shadow_ids), self.feat_dim
            if K < D: self._U_cache[key] = np.zeros((K,D), dtype=np.float32)
            else:
                R = self.rng.standard_normal((K, D)).astype(np.float32)
                Q, _ = np.linalg.qr(R, mode='reduced')
                self._U_cache[key] = Q[:, :D]
        return self._U_cache[key]

    def encode_bulk(self, feat: np.ndarray, shadow_ids: np.ndarray, slice_id: int):
        U = self._U(shadow_ids)
        y = U @ feat
        payload = {"f": y.astype(np.float32).tolist()}
        for nid, val in zip(shadow_ids, payload["f"]):
            self.store[(int(nid), int(slice_id))] = float(val)
        return payload

    def decode_boundary(self, shadow_ids: np.ndarray, slice_id: int, payload: dict) -> np.ndarray:
        U = self._U(shadow_ids)
        y = np.array(payload.get("f", []), dtype=np.float32)
        if y.size == 0:
            return np.zeros(self.feat_dim, dtype=np.float32)
        y = y[:U.shape[0]]
        return (U.T @ y).astype(np.float32)

    def encode(self, embedding: np.ndarray) -> np.ndarray:
        """Compress a high-D embedding to feat_dim (default 8). Safe fallbacks if compressor isn't ready."""
        try:
            v = np.asarray(embedding, dtype=np.float32).reshape(-1)
            comp = getattr(self, "_compressor", None)
            if comp is None:
                in_dim = int(v.size)
                try:
                    comp = TinyCompressor(in_dim=in_dim, code_dim=int(self.feat_dim))
                except Exception:
                    comp = None
                setattr(self, "_compressor", comp)
            if comp is not None:
                try:
                    z = comp.encode(v)
                except Exception:
                    z = None
            else:
                z = None
            if z is None:
                z = v[: self.feat_dim]
            z = np.asarray(z, dtype=np.float32).reshape(-1)
            if z.size < self.feat_dim:
                z = np.pad(z, (0, self.feat_dim - z.size))
            elif z.size > self.feat_dim:
                z = z[: self.feat_dim]
            return z.astype(np.float32)
        except Exception:
            v = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if v.size < self.feat_dim:
                v = np.pad(v, (0, self.feat_dim - v.size))
            return v[: self.feat_dim].astype(np.float32)

class EntropyMap:
    def __init__(self, fabric: "E8BoundaryFabric", k_bits_per_edge: float = 4.0):
        self.fabric, self.k = fabric, float(k_bits_per_edge)
        self.A = (fabric.A > 0).astype(np.float32)
        self.N = int(self.A.shape[0])

    def perimeter(self, region_nodes: np.ndarray) -> float:
        mask = np.zeros(self.N, dtype=np.float32)
        mask[region_nodes] = 1.0
        cut = np.sum(self.A[region_nodes], axis=0) * (1.0 - mask)
        return float(cut.sum())

    def budget_bits(self, region_nodes: np.ndarray) -> float:
        return self.k * self.perimeter(region_nodes)

    def usage_bits(self, store: dict, region_nodes: np.ndarray, slice_id: Optional[int] = None) -> float:
        rset = set(int(i) for i in region_nodes.tolist())
        bits = 0.0
        for (nid, sid), val in store.items():
            if nid in rset and (slice_id is None or sid == int(slice_id)):
                bits += 32.0
        return float(bits)

    def deficit_ratio(self, store: dict, region_nodes: np.ndarray, slice_id: Optional[int] = None) -> float:
        B = self.budget_bits(region_nodes) + 1e-6
        U = self.usage_bits(store, region_nodes, slice_id)
        return float((U - B) / B)

class SensorProjector:
    def __init__(self, in_dim, out_dim=8, seed=None):
        self.in_dim, self.out_dim = in_dim, out_dim
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.standard_normal((in_dim, out_dim)).astype(np.float32) * 0.1
        self.mu = np.zeros(in_dim, dtype=np.float32)

    def pca_bootstrap(self, embeddings: np.ndarray, top_k=240):
        if embeddings.shape[0] < self.out_dim or PCA is None: return
        try:
            pca = PCA(n_components=self.out_dim)
            pca.fit(embeddings[:top_k])
            self.W, self.mu = pca.components_.T, pca.mean_
            # Use basic print since console might not be available
            print(f"[PROJ] Bootstrapped with PCA on {top_k} embeddings.")
        except Exception as e:
            print(f"[PROJ] PCA bootstrap failed: {e}. Falling back to random init.")

    def project(self, embedding):
        if embedding.shape[0] != self.in_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.in_dim}, got {embedding.shape[0]}.")
        return normalize_vector((embedding - self.mu) @ self.W)

    def train(self, embeddings, labels, roots_unit, epochs=3, lr=5e-3, batch_size=64, **kwargs):
        if embeddings.shape[0] < batch_size: return
        print(f"[PROJ] Starting training burst on {embeddings.shape[0]} samples.")
        for _ in range(epochs):
            indices = self.rng.integers(0, embeddings.shape[0], size=batch_size)
            for i in indices:
                e, y = embeddings[i], labels[i]
                s = normalize_vector((e - self.mu) @ self.W)
                delta_W = lr * np.outer(e - self.mu, roots_unit[y] - s)
                self.W += delta_W


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
