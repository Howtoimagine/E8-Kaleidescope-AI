"""Quantum lattice / graph exploratory engine (modular extraction).

Extracted components:
- QuantumConfig
- _safe_node_to_root_idx (memory -> root index helper)
- build_adjacency_240_from_memory (graph-derived adjacency)
- QuantumEngine (adaptive stepping, measurement strategies, telemetry)

External integration points (mind, memory, autoencoder) are optional; the
engine degrades gracefully if unavailable.
"""
from __future__ import annotations

import os
import numpy as np
from typing import Any, Optional, Dict, List

# Optional scientific stack
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False

try:
    from scipy.sparse import csr_matrix, diags  # type: ignore
    from scipy.sparse.linalg import expm_multiply  # type: ignore
except Exception:  # pragma: no cover
    csr_matrix = None  # type: ignore
    diags = None  # type: ignore
    expm_multiply = None  # type: ignore

# Console protocol (duck-typed)
class _ConsoleLike:
    def log(self, *a, **k): ...


def safe_tensor_to_numpy(t):  # minimalist helper (avoid circular util import)
    try:
        return t.detach().cpu().numpy()  # type: ignore
    except Exception:
        return None


class QuantumConfig:
    def __init__(self, gamma: float = 0.03, dt: float = 0.25, batch: int = 9,
                 dephase: float = 0.0, locality_sigma: float = 1.5,
                 seed=None, topk_amp: int = 5, non_linearity_strength: float = 2.5):
        self.gamma = gamma
        self.mode = os.getenv('E8_QMODE', 'lattice')
        self.lambda_V = float(os.getenv('E8_Q_LAMBDA', '0.2'))
        self.decay_tau = float(os.getenv('E8_Q_DECAY_TAU', '600'))
        self.rebuild_every = int(os.getenv('E8_Q_REBUILD_EVERY', '10'))
        self.alpha_edge = float(os.getenv('E8_Q_ALPHA_EDGE', '1.0'))
        self.reward_gain = float(os.getenv('E8_Q_REWARD_GAIN', '0.5'))
        self.dt = dt
        self.batch = batch
        self.dephase = dephase
        self.locality_sigma = locality_sigma
        self.seed = seed
        self.topk_amp = topk_amp
        self.non_linearity_strength = non_linearity_strength


def _safe_node_to_root_idx(nid, memory, mind):
    try:
        node = memory.graph_db.get_node(nid)  # type: ignore
        loc = node.get('blueprint_location_id') if node else None
        if isinstance(loc, (int, np.integer)) and 0 <= int(loc) < 240:
            return int(loc)
        vec = memory.main_vectors.get(nid)  # type: ignore
        if vec is not None and TORCH_AVAILABLE and getattr(mind, 'autoencoder', None) and mind.autoencoder and getattr(mind.autoencoder, 'is_trained', False):
            with torch.no_grad():  # type: ignore
                arr = np.asarray(vec, dtype=np.float32)
                z8 = mind.autoencoder.project_to_dim(torch.from_numpy(arr).unsqueeze(0), 8)  # type: ignore
                if z8 is not None:
                    root_idx = mind.physics.find_nearest_root_index(safe_tensor_to_numpy(z8.squeeze(0)))  # type: ignore
                    return int(root_idx) if root_idx is not None else None
    except Exception:
        return None
    return None


def build_adjacency_240_from_memory(memory, mind, alpha=1.0, decay_tau=600.0, reward_gain=0.5):
    from math import exp
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    try:
        G = memory.graph_db.graph  # type: ignore
        now = getattr(mind, 'step_num', 0)
        for u, v, attr in G.edges(data=True):  # type: ignore
            iu = _safe_node_to_root_idx(u, memory, mind)
            iv = _safe_node_to_root_idx(v, memory, mind)
            if iu is None or iv is None or iu == iv:
                continue
            w = float(attr.get('weight', 1.0))
            ts = float(attr.get('ts', 0.0))
            lu = float((G.nodes[u] or {}).get('last_step', 0)) if u in G.nodes else 0.0
            lv = float((G.nodes[v] or {}).get('last_step', 0)) if v in G.nodes else 0.0
            last_seen = max(ts, lu, lv)
            rec = float(exp(-(max(0.0, now - last_seen)) / max(1e-6, decay_tau)))
            ru = float((G.nodes[u] or {}).get('insight_reward', 0.0)) if u in G.nodes else 0.0
            rv = float((G.nodes[v] or {}).get('insight_reward', 0.0)) if v in G.nodes else 0.0
            rewd = float(attr.get('reward', 0.0))
            rbar = (ru + rv + rewd) / 3.0
            wij = alpha * w * rec * (1.0 + reward_gain * rbar)
            if wij <= 0:
                continue
            rows.append(iu); cols.append(iv); data.append(wij)
            rows.append(iv); cols.append(iu); data.append(wij)
        if csr_matrix is None:
            return None
        import numpy as _np
        if data:
            A = csr_matrix((_np.asarray(data, dtype=_np.float32), (_np.asarray(rows), _np.asarray(cols))), shape=(240, 240))
        else:
            A = csr_matrix((240, 240), dtype=_np.float32)
        return A
    except Exception:
        if csr_matrix is None:
            return None
        import numpy as _np
        return csr_matrix((240, 240), dtype=_np.float32)


class QuantumEngine:
    def __init__(self, physics, config: QuantumConfig, console: _ConsoleLike):
        # Mind instance (attached later) provides memory + step counter
        self.mind: Any | None = None
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
        try:
            self.console.log("⚛️ [INIT] Quantum Engine online (Non-Linear Edition).")
        except Exception:
            pass

    def attach_mind(self, mind_instance):
        self.mind = mind_instance
        return self

    def build_hamiltonian(self, V: Optional[np.ndarray] = None):
        if diags is None or csr_matrix is None:
            return
        if V is None:
            V = np.zeros(240, dtype=np.float32)
        use_graph = (getattr(self.config, 'mode', 'lattice') == 'graph') and (getattr(self, 'mind', None) is not None)
        if use_graph and (self._last_graph_build_step < 0 or (getattr(self.mind, 'step_num', 0) - self._last_graph_build_step) >= getattr(self.config, 'rebuild_every', 10)):
            mind_memory = getattr(self.mind, 'memory', None) if self.mind is not None else None
            if mind_memory is not None:
                A = build_adjacency_240_from_memory(mind_memory, self.mind,
                                                    alpha=getattr(self.config, 'alpha_edge', 1.0),
                                                    decay_tau=getattr(self.config, 'decay_tau', 600.0),
                                                    reward_gain=getattr(self.config, 'reward_gain', 0.5))
            else:
                A = None
            if A is not None:
                H = (-self.config.gamma * A.astype(np.complex64)) + getattr(self.config, 'lambda_V', 0.2) * diags(V)
                self.H = csr_matrix(H)
                self._last_graph_build_step = getattr(self.mind, 'step_num', 0) if self.mind is not None else 0
        else:
            H = (self.config.gamma * self.physics.L_norm.astype(np.complex64)) + diags(V)
            self.H = csr_matrix(H)
        self._last_H = self.H
        self._last_potential = np.asarray(V).copy()

    def _probs(self):
        p = np.abs(self.psi) ** 2
        return p / (np.sum(p, axis=1, keepdims=True) + 1e-12)

    def step_adaptive(self, tv_target=0.07, dt_min=0.02, dt_max=1.2):
        if self.H is None or expm_multiply is None:
            return 0.0
        p0 = self._probs()
        H_eff = self.H.copy()
        if self.config.non_linearity_strength != 0:
            feedback = self.config.non_linearity_strength * p0[0]
            H_eff += diags(feedback.astype(np.float32), 0)  # type: ignore
        psi_new = expm_multiply(-1j * H_eff * self.config.dt, self.psi.T).T  # type: ignore
        nrm = np.linalg.norm(psi_new, axis=1, keepdims=True)
        self.psi = psi_new / np.maximum(nrm, 1e-12)
        p1 = self._probs()
        tv = 0.5 * float(np.abs(p0 - p1).sum(axis=1).mean())
        if tv < 0.5 * tv_target:
            self.config.dt = min(dt_max, self.config.dt * 1.25)
        elif tv > 1.5 * tv_target:
            self.config.dt = max(dt_min, self.config.dt * 0.66)
        if self.config.dephase > 0:
            mag = np.abs(self.psi)
            self.psi = (1.0 - self.config.dephase) * self.psi + self.config.dephase * mag
            nrm = np.linalg.norm(self.psi, axis=1, keepdims=True)
            self.psi /= np.maximum(nrm, 1e-12)
        try:
            self._last_norm = float(np.mean(np.sum(np.abs(self.psi) ** 2, axis=1)))
            Href = self._last_H
            if Href is not None and getattr(Href, 'ndim', 0) == 2:
                Energies = []
                for b in range(self.psi.shape[0]):
                    v = self.psi[b].reshape(-1, 1)
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
        if isinstance(prev_idx, (list, np.ndarray)):
            masks = np.stack([self.physics.heat_mask_cached(i, sigma) for i in prev_idx])
        else:
            masks = np.tile(self.physics.heat_mask_cached(int(prev_idx), sigma), (self.config.batch, 1))
        P *= masks
        P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
        return np.array([self.rng.choice(P.shape[1], p=p) for p in P], dtype=np.int32)

    def measure_hybrid(self, prev_idx=None, sigma=None, topk=None):
        if prev_idx is None:
            prev_idx = 0
        if not hasattr(self, 'psi'):
            return self.measure_local([prev_idx] * self.config.batch, sigma)
        B, N = self.psi.shape
        P = np.abs(self.psi) ** 2
        P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)
        Vlast = self._last_potential
        if Vlast is not None and np.size(Vlast) == N:
            soft = np.maximum(0.0, -np.real(np.asarray(Vlast).reshape(1, -1)))
            if topk is None:
                topk = int(getattr(self.config, 'topk_amp', 5) or 5)
            idx = np.argpartition(soft[0], -topk)[-topk:]
            mask = np.zeros_like(P)
            mask[:, idx] = 1.0
            Amp = np.sqrt(P) * np.sqrt(soft + 1e-12)
            P = (Amp ** 2) * mask
            P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)
        if sigma is None:
            sigma = float(getattr(self.config, 'locality_sigma', 1.5) or 1.5)
        hops = self.physics.hops
        w = np.exp(-(hops[prev_idx] ** 2) / (2.0 * sigma * sigma))
        P = P * w.reshape(1, -1)
        P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)
        choices = [int(self.rng.choice(N, p=P[b])) for b in range(B)]
        return choices

    def telemetry_state(self) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            'norm': float(self._last_norm),
            'energy': float(self._last_energy),
            'topk_amp': int(getattr(self.config, 'topk_amp', 5)),
            'locality_sigma': float(getattr(self.config, 'locality_sigma', 1.5)),
        }
        base['quantum_telemetry'] = {
            'dt': float(getattr(self.config, 'dt', 0.01)),
            'gamma': float(getattr(self.config, 'gamma', 1.0)),
            'dephase': float(getattr(self.config, 'dephase', 0.0)),
            'lam': float(getattr(self, '_last_lam', float('nan'))),
            'psi_entropy': float(getattr(self, '_last_psi_entropy', float('nan'))),
        }
        base['novelty'] = float(getattr(self, '_last_novelty', float('nan')))
        base['compression_gain'] = float(getattr(self, '_last_compression_gain', getattr(self, '_last_compression', float('nan'))))
        base['disagreement'] = float(getattr(self, '_last_disagreement', float('nan')))
        base['kdtree_failures'] = int(getattr(self, '_last_kdtree_failures', -1))
        return base

    def measure_ablation(self, prev_idx: int, sigma: Optional[float] = None, window: int = 5, trials: int = 512) -> Dict[str, Any]:
        if sigma is None:
            sigma = float(getattr(self.config, 'locality_sigma', 1.5) or 1.5)
        B, N = getattr(self, 'psi', np.zeros((1, 1))).shape
        total = trials * B
        if total == 0:
            return {}
        local_choices: List[int] = []
        hybrid_choices: List[int] = []
        for _ in range(trials):
            local_choices.extend(self.measure_local([prev_idx] * B, sigma=sigma))
            hybrid_choices.extend(self.measure_hybrid(prev_idx=prev_idx, sigma=sigma))
        local_counts = np.bincount(np.asarray(local_choices), minlength=N)
        hybrid_counts = np.bincount(np.asarray(hybrid_choices), minlength=N)
        lo = max(0, prev_idx - window); hi = min(N - 1, prev_idx + window)
        local_win = int(local_counts[lo:hi + 1].sum())
        hybrid_win = int(hybrid_counts[lo:hi + 1].sum())
        return {
            'prev_idx': int(prev_idx),
            'window': int(window),
            'sigma': float(sigma),
            'trials': int(trials),
            'batch': int(B),
            'N': int(N),
            'local_win': local_win,
            'hybrid_win': hybrid_win,
            'local_rate': float(local_win / total),
            'hybrid_rate': float(hybrid_win / total),
        }

__all__ = [
    'QuantumConfig', 'QuantumEngine', 'build_adjacency_240_from_memory'
]
