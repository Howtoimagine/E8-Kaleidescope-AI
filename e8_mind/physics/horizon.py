"""Horizon boundary structures and coupling kernels.

Extracted from legacy monolith (M24.1) and modularized.
Provides:
- HorizonLayer: container for a boundary/horizon surface
- HorizonManager: aggregates E8 and shell horizons, snapshot utilities
- build_e8_horizon: construct E8 blueprint horizon from projected tetra edges
- build_cross_horizon_kernel: sparse coupling between E8 horizon and shell horizon

Env flags (future extension):
  E8_HORIZON_SIGMA, E8_HORIZON_BC, E8_USE_HORIZONS
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Any

import numpy as np

try:  # optional deps
    from scipy.sparse import coo_matrix
except Exception:  # pragma: no cover
    coo_matrix = None  # type: ignore

try:  # optional for build_e8_horizon
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:  # pragma: no cover
    NearestNeighbors = None  # type: ignore

# Console type hint (lightweight)
class _ConsoleLike:
    def log(self, *a, **k): ...  # noqa: D401


class HorizonLayer:
    """Generic boundary layer container."""
    def __init__(self, name: str):
        self.name = name
        self.pos: Optional[np.ndarray] = None       # (N,3) or (N,D)
        self.normals: Optional[np.ndarray] = None   # (N,3) unit outward normals
        self.indices: Optional[np.ndarray] = None   # indices of lattice nodes
        self.meta: Dict[str, Any] = {}


class HorizonManager:
    """Holds H_E8 and per-shell H_shell[d] plus cross-horizon kernels C_d."""
    def __init__(self):
        self.H_E8: Optional[HorizonLayer] = None
        self.H_shell: Dict[int, HorizonLayer] = {}
        self.C_d: Dict[int, Any] = {}  # (Nh_shell, Nh_e8) sparse coupling matrices

    def get_horizon_layers(self) -> Dict[str, HorizonLayer]:
        layers: Dict[str, HorizonLayer] = {}
        if self.H_E8:
            layers['H_E8'] = self.H_E8
        for d, h in self.H_shell.items():
            layers[f'H_shell_{d}'] = h
        return layers

    def get_horizon_snapshot(self) -> Optional[Dict[str, Any]]:
        """Create snapshot of current horizon state (lightweight metrics)."""
        try:
            if not self.H_E8 and not self.H_shell:
                return None
            snapshot: Dict[str, Any] = {
                'timestamp': time.time(),
                'layers': {},
                'energy_totals': {},
                'field_couplings': {}
            }
            # E8 horizon
            if self.H_E8:
                snapshot['layers']['H_E8'] = {
                    'name': self.H_E8.name,
                    'num_sites': int(len(self.H_E8.indices)) if self.H_E8.indices is not None else 0,
                    'dimensions': int(self.H_E8.pos.shape[1]) if self.H_E8.pos is not None else 0,
                    'has_normals': self.H_E8.normals is not None,
                    'meta': self.H_E8.meta,
                }
            # Shell horizons
            for dim, h_layer in self.H_shell.items():
                snapshot['layers'][f'H_shell_{dim}'] = {
                    'name': h_layer.name,
                    'num_sites': int(len(h_layer.indices)) if h_layer.indices is not None else 0,
                    'dimensions': int(h_layer.pos.shape[1]) if h_layer.pos is not None else 0,
                    'has_normals': h_layer.normals is not None,
                    'meta': h_layer.meta,
                }
            # Placeholder energy model
            total_energy = 0.0
            boundary_energy = 0.0
            stress_energy = 0.0
            for layer_name, layer_info in snapshot['layers'].items():
                num_sites = layer_info['num_sites']
                dimensions = layer_info['dimensions'] or 3
                layer_energy = num_sites * dimensions * 0.1
                total_energy += layer_energy
                if 'E8' in layer_name:
                    boundary_energy += layer_energy * 1.5
                else:
                    boundary_energy += layer_energy
                stress_energy += layer_energy * 0.3
            snapshot['energy_totals'] = {
                'total_energy': float(total_energy),
                'boundary_energy': float(boundary_energy),
                'stress_energy': float(stress_energy),
            }
            # Couplings metadata
            for dim, coupling in self.C_d.items():
                if hasattr(coupling, 'shape'):
                    nnz = getattr(coupling, 'nnz', 0)
                    snapshot['field_couplings'][f'C_{dim}'] = {
                        'shape': tuple(getattr(coupling, 'shape', (0, 0))),
                        'nnz': int(nnz),
                    }
            return snapshot
        except Exception:
            return None


def build_e8_horizon(physics, blueprint_positions_3d, tetra_edges, console: Optional[_ConsoleLike] = None) -> HorizonLayer:
    """Construct H_E8: boundary points at projected E8 tetra centers/edges.

    Parameters
    ----------
    physics : Any
        Physics context (unused placeholder for future coupling).
    blueprint_positions_3d : array-like, (N,3)
        Unique 3D projected E8 root positions.
    tetra_edges : list[tuple[int,int]]
        Edges referencing indices in blueprint_positions_3d.
    console : logger-like, optional
        For diagnostic logging.
    """
    if NearestNeighbors is None:
        raise RuntimeError("scikit-learn required for build_e8_horizon")
    P = np.asarray(blueprint_positions_3d, dtype=np.float64)
    H = HorizonLayer("H_E8")
    edge_mid = []
    edge_norm = []
    nbrs = NearestNeighbors(n_neighbors=min(12, len(P))).fit(P)
    for (i, j) in tetra_edges:
        m = 0.5 * (P[i] + P[j])
        _, idx = nbrs.kneighbors(m.reshape(1, -1))
        Q = P[idx[0]]
        C = np.cov((Q - Q.mean(0)).T)
        w, v = np.linalg.eigh(C)
        n = v[:, 0]
        n = n / (np.linalg.norm(n) + 1e-12)
        edge_mid.append(m)
        edge_norm.append(n)
    H.pos = np.vstack(edge_mid) if edge_mid else np.zeros((0, 3))
    H.normals = np.vstack(edge_norm) if edge_norm else np.zeros((0, 3))
    H.indices = None
    H.meta['edges'] = tetra_edges
    if console:
        console.log(f"[Horizon] Built H_E8 with {len(edge_mid)} edge sites.")
    return H


def build_cross_horizon_kernel(H_e8: HorizonLayer, H_shell: HorizonLayer, symmetry_weight: float = 1.0):
    """Build sparse C_d mapping H_E8 -> H_shell by lifted adjacency & distance.

    Returns a scipy.sparse.coo_matrix mapping shell -> e8 horizon sites.
    If SciPy unavailable, returns a placeholder dict with weight triplets.
    """
    if coo_matrix is None:
        return {'rows': [], 'cols': [], 'vals': [], 'shape': (0, 0)}
    if H_e8.pos is None or H_shell.pos is None or H_e8.pos.size == 0 or H_shell.pos.size == 0:
        return coo_matrix((0, 0))
    A = H_shell.pos.astype(np.float64)
    B = H_e8.pos.astype(np.float64)[:, :A.shape[1]]
    rows, cols, vals = [], [], []
    for i, a in enumerate(A):
        d2 = np.sum((B - a) ** 2, axis=1)
        idx = np.argsort(d2)[:8]
        w = np.exp(-d2[idx] / (np.median(d2[idx]) + 1e-12)) * symmetry_weight
        s = w.sum() + 1e-12
        for j, wij in zip(idx, w):
            rows.append(i)
            cols.append(int(j))
            vals.append(float(wij / s))
    return coo_matrix((vals, (rows, cols)), shape=(A.shape[0], B.shape[0]))

__all__ = [
    'HorizonLayer', 'HorizonManager', 'build_e8_horizon', 'build_cross_horizon_kernel'
]
