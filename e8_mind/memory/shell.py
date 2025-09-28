from __future__ import annotations

from typing import Any, Dict, Optional, List
import os
import numpy as np

try:
    import clifford  # type: ignore
    CLIFFORD_AVAILABLE = True
except Exception:
    CLIFFORD_AVAILABLE = False


class DimensionalShell:
    def __init__(self, dim: int, mind_instance: Any):
        self.dim = int(dim)
        self.mind = mind_instance
        self.vectors: Dict[str, Any] = {}
        self.vector_mode = "clifford"
        self.rotor_generator = None
        self.orientation = 1

        if CLIFFORD_AVAILABLE:
            try:
                self.layout, self.blades = clifford.Cl(self.dim)
                self.basis_vectors = [self.blades.get(f"e{i+1}") for i in range(self.dim)]
                # Try to import a rotor generator from physics.rotor if available
                try:
                    from e8_mind.physics.rotor import CliffordRotorGenerator  # type: ignore
                    self.rotor_generator = CliffordRotorGenerator(mind_instance, self.layout, self.blades)
                except Exception:
                    self.rotor_generator = None
                self.orientation = getattr(self.layout, "scalar", 1)
            except Exception:
                # degrade to numpy
                self.layout, self.blades = None, {}
                self.basis_vectors = [np.eye(self.dim, dtype=float)[i] for i in range(self.dim)]
                self.vector_mode = "numpy"
        else:
            self.layout, self.blades = None, {}
            self.basis_vectors = [np.eye(self.dim, dtype=float)[i] for i in range(self.dim)]
            self.vector_mode = "numpy"

        # Bivector basis (Clifford mode only)
        self.bivector_basis: list[Any] = []
        if CLIFFORD_AVAILABLE and self.vector_mode == "clifford":
            self._build_bivector_basis()

    def _ensure_numpy_mode(self):
        if self.vector_mode != "numpy":
            self.vector_mode = "numpy"
            new_vectors = {}
            for nid, mv in self.vectors.items():
                try:
                    coeffs = [float(mv[bv]) for bv in self.basis_vectors]
                    new_vectors[nid] = np.array(coeffs, dtype=np.float32)
                except Exception:
                    pass
            if new_vectors:
                self.vectors.update(new_vectors)

    def add_vector(self, node_id: str, vector: np.ndarray):
        if vector is None:
            return
        if hasattr(vector, "shape") and vector.shape[0] != self.dim:
            padded_vector = np.zeros(self.dim, dtype=np.float32)
            size_to_copy = min(vector.shape[0], self.dim)
            padded_vector[:size_to_copy] = vector[:size_to_copy]
            vector = padded_vector

        # Snap to lattice through manager mixin if present
        try:
            snapped_vector = self.mind._snap_to_lattice(vector, self.dim)
        except Exception:
            v = np.asarray(vector, dtype=np.float32).reshape(-1)
            if v.size != self.dim:
                out = np.zeros(self.dim, dtype=np.float32)
                n = min(v.size, self.dim)
                out[:n] = v[:n]
                v = out
            nrm = float(np.linalg.norm(v))
            snapped_vector = (v / nrm).astype(np.float32) if nrm > 1e-6 else v.astype(np.float32)

        if self.vector_mode == "clifford":
            try:
                mv = 0
                for val, bv in zip(snapped_vector, self.basis_vectors):
                    mv = mv + float(val) * bv
                self.vectors[node_id] = mv
                return
            except Exception:
                self._ensure_numpy_mode()

        self.vectors[node_id] = np.asarray(snapped_vector, dtype=np.float32)

    def get_vector(self, node_id: str) -> Optional[np.ndarray]:
        v = self.vectors.get(node_id)
        if v is None:
            return None
        if self.vector_mode == "clifford":
            try:
                return np.array([float(v[bv]) for bv in self.basis_vectors], dtype=np.float32)
            except Exception:
                self._ensure_numpy_mode()
        return np.asarray(v, dtype=np.float32)

    def get_all_vectors_as_matrix(self) -> tuple[Optional[np.ndarray], list[str]]:
        if not self.vectors:
            return None, None  # type: ignore
        node_ids = list(self.vectors.keys())
        if self.vector_mode == "clifford":
            try:
                matrix_list = [[float(self.vectors[nid][bv]) for bv in self.basis_vectors] for nid in node_ids]
                return np.array(matrix_list, dtype=np.float32), node_ids
            except Exception:
                self._ensure_numpy_mode()
        matrix = np.stack([np.asarray(self.vectors[nid], dtype=np.float32) for nid in node_ids], axis=0)
        return matrix, node_ids

    def _build_bivector_basis(self):
        if not CLIFFORD_AVAILABLE or self.vector_mode != "clifford":
            self.bivector_basis = []
            return
        try:
            self.bivector_basis = []
            for i in range(self.dim):
                for j in range(i + 1, self.dim):
                    self.bivector_basis.append(self.basis_vectors[i] ^ self.basis_vectors[j])
        except Exception:
            self.bivector_basis = []

    def spin_with_bivector(self, bivector_coeffs, angle):
        if self.vector_mode != "clifford" or not CLIFFORD_AVAILABLE or not self.vectors:
            return
        try:
            if not hasattr(self, "bivector_basis") or not self.bivector_basis:
                self._build_bivector_basis()
            B = 0
            k = min(len(self.bivector_basis), len(bivector_coeffs))
            for idx in range(k):
                try:
                    B = B + float(bivector_coeffs[idx]) * self.bivector_basis[idx]
                except Exception:
                    pass
            Bn = B.normal() if hasattr(B, "normal") else None
            if Bn is None or not hasattr(self.layout, "multi_vector"):
                return
            R = np.cos(angle / 2.0) - np.sin(angle / 2.0) * Bn
            for nid in list(self.vectors.keys()):
                try:
                    mv = self.vectors[nid]
                    self.vectors[nid] = R * mv * (~R) if hasattr(R, "__invert__") else mv
                except Exception:
                    pass
        except Exception:
            self._ensure_numpy_mode()

    # Minimal boundary extraction shim used by ProximityEngine horizon-aware helpers
    def extract_shell_horizon(self, degree_cap=6, voronoi_quantile=0.25):
        from e8_mind.physics.horizon import HorizonLayer  # lazy import
        H = HorizonLayer(f"H_shell[{self.dim}]")
        boundary_idx = []
        areas = []
        for nid, data in getattr(self, 'shell_nodes', {}).items():
            deg = len(data.get("neighbors", []))
            areas.append(data.get("voronoi_area", 1.0))
            if deg <= degree_cap:
                boundary_idx.append(nid)
        pos = []
        normals = []
        for nid in boundary_idx:
            d = getattr(self, 'shell_nodes', {}).get(nid, {})
            p = np.asarray(d.get("pos"), dtype=float)
            neigh = d.get("neighbors", [])
            if not len(neigh):
                continue
            N = np.stack([np.asarray(getattr(self, 'shell_nodes', {}).get(j, {}).get("pos"), dtype=float) for j in neigh], axis=0)
            center = N.mean(0)
            n = (p - center)
            n /= (np.linalg.norm(n) + 1e-12)
            pos.append(p)
            normals.append(n[:3] if n.shape[0] >= 3 else np.pad(n, (0, 3 - n.shape[0])))
        H.indices = boundary_idx
        H.pos = np.stack(pos, axis=0) if pos else np.zeros((0, self.dim))
        H.normals = np.stack(normals, axis=0) if normals else np.zeros((0, 3))
        H.meta["degree_cap"] = degree_cap
        H.meta["voronoi_q"] = voronoi_quantile
        try:
            if getattr(self, 'mind', None) and getattr(self.mind, 'console', None):
                self.mind.console.log(f"[Horizon] Shell d={self.dim}: {len(boundary_idx)} boundary sites.")
        except Exception:
            pass
        return H
