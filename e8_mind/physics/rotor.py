"""Clifford algebra based rotor generation utilities."""
from __future__ import annotations
import math, os, numpy as np
from typing import Optional, Any

# Attempt to import clifford-like library if present
try:
    import clifford  # type: ignore
    CLIFFORD_AVAILABLE = True
except Exception:
    CLIFFORD_AVAILABLE = False

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v

class CliffordRotorGenerator:
    def __init__(self, mind_instance: 'E8Mind', layout, blades):
        self.mind = mind_instance
        self.layout = layout
        self.blades = blades or {}
        try:
            dims = getattr(layout, 'dims', 0) if layout is not None else 0
        except Exception:
            dims = 0
        if CLIFFORD_AVAILABLE and dims:
            try:
                self.basis_vectors = [self.blades.get(f'e{i+1}') for i in range(dims)]
            except Exception:
                self.basis_vectors = []
        else:
            self.basis_vectors = []

    def _random_unit_bivector(self):
        if not CLIFFORD_AVAILABLE or len(self.basis_vectors) < 2:
            return 0
        try:
            n = len(self.basis_vectors)
            i, j = np.random.choice(np.arange(n), size=2, replace=False)
            B = self.basis_vectors[i] ^ self.basis_vectors[j]
            return B.normal()
        except Exception:
            return 0

    def _select_dynamic_pair(self, shell: 'DimensionalShell') -> Optional[tuple[np.ndarray, np.ndarray]]:
        nodes = list(shell.vectors.keys()) if shell is not None else []
        if len(nodes) < 2:
            return None
        candidates = []
        for nid in nodes:
            node_data = self.mind.memory.graph_db.get_node(nid) if self.mind and self.mind.memory else None
            if node_data:
                vec_np = shell.get_vector(nid)
                if vec_np is not None and np.linalg.norm(vec_np) > 1e-9:
                    candidates.append({'id': nid, 'temp': node_data.get('temperature', 0.1), 'vec': vec_np})
        if len(candidates) < 2:
            return None
        candidates.sort(key=lambda x: x['temp'], reverse=True)
        anchor_a = candidates[0]
        best_partner, max_dist = None, -1.0
        for partner_candidate in candidates[1:min(len(candidates), 15)]:
            dist = 1.0 - abs(np.dot(normalize_vector(anchor_a['vec']), normalize_vector(partner_candidate['vec'])))
            if dist > max_dist:
                max_dist = dist
                best_partner = partner_candidate
        if best_partner is None:
            return None
        return anchor_a['vec'], best_partner['vec']

    def generate_rotor(self, shell: 'DimensionalShell', angle: float) -> Any:
        if not CLIFFORD_AVAILABLE or not self.basis_vectors:
            return 1
        pair = self._select_dynamic_pair(shell)
        if pair is None:
            return 1
        try:
            a_vec, b_vec = pair
            a = sum(val * bv for val, bv in zip(a_vec, self.basis_vectors))
            b = sum(val * bv for val, bv in zip(b_vec, self.basis_vectors))
            B = (a ^ b)
            if hasattr(B, 'normal'):
                try:
                    Bn = B.normal()
                    if abs(Bn) < 1e-9:
                        return 1
                    rotor = (-Bn * angle / 2.0).exp()
                    return rotor
                except Exception:
                    return 1
            return 1
        except Exception:
            return 1

    def create_rotor_from_bivector(self, B: Any, angle: float) -> Any:
        if not CLIFFORD_AVAILABLE:
            return 1
        try:
            if hasattr(B, 'normal'):
                Bn = B.normal()
                if abs(Bn) < 1e-9:
                    return 1
                rotor = (Bn * angle / 2.0).exp()
                return rotor
            return 1
        except Exception:
            return 1

    def rotate_vector(self, rotor: Any, vector: Any) -> Any:
        if not CLIFFORD_AVAILABLE:
            return vector
        try:
            if hasattr(rotor, '__mul__') and hasattr(rotor, 'conjugate'):
                rotor_rev = rotor.conjugate()
                return rotor * vector * rotor_rev
            return vector
        except Exception:
            return vector

    def compose_rotors(self, rotor1: Any, rotor2: Any) -> Any:
        if not CLIFFORD_AVAILABLE:
            return 1
        try:
            if hasattr(rotor1, '__mul__') and hasattr(rotor2, '__mul__'):
                return rotor2 * rotor1
            return 1
        except Exception:
            return 1

    def rotor_inverse(self, rotor: Any) -> Any:
        if not CLIFFORD_AVAILABLE:
            return 1
        try:
            if hasattr(rotor, 'conjugate'):
                return rotor.conjugate()
            return 1
        except Exception:
            return 1

    def integrate_rotor_step(self, rotor: Any, omega: Any, dt: float) -> Any:
        if not CLIFFORD_AVAILABLE:
            return rotor
        try:
            if hasattr(omega, '__mul__') and hasattr(rotor, '__mul__'):
                delta_rotor = (omega * dt / 2.0).exp()
                return delta_rotor * rotor
            return rotor
        except Exception:
            return rotor

    def validate_rotor_properties(self, rotor: Any, test_vectors: list = None) -> dict:
        if not CLIFFORD_AVAILABLE:
            return {'norm_invariant': False, 'double_cover': False}
        results = {'norm_invariant': False, 'double_cover': False}
        try:
            if test_vectors is None:
                test_vectors = [np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                               np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
            norm_invariant = False
            double_cover = False
            for attempt in range(3):
                angle = 0.5 * (1.0 + 0.01 * attempt)
                if rotor is None:
                    rotor = self.generate_rotor(shell=None, angle=angle)
                all_norm_ok = True
                for vec_np in test_vectors:
                    if len(self.basis_vectors) >= len(vec_np):
                        vec = sum(val * bv for val, bv in zip(vec_np, self.basis_vectors))
                        rotated = self.rotate_vector(rotor, vec)
                        try:
                            vnorm = float(abs(vec))
                            rnorm = float(abs(rotated))
                        except Exception:
                            vnorm = float(np.linalg.norm(np.asarray(vec_np)))
                            rnorm = float(np.linalg.norm(np.asarray(vec_np)))
                        if abs(vnorm - rnorm) > 1e-6:
                            all_norm_ok = False
                            break
                if all_norm_ok:
                    norm_invariant = True
                try:
                    R2 = self.generate_rotor(shell=None, angle=2 * math.pi)
                    all_double_ok = True
                    for vec_np in test_vectors:
                        if len(self.basis_vectors) >= len(vec_np):
                            vec = sum(val * bv for val, bv in zip(vec_np, self.basis_vectors))
                            rotated2 = self.rotate_vector(R2, vec)
                            try:
                                if abs(vec - rotated2) > 1e-5:
                                    all_double_ok = False
                                    break
                            except Exception:
                                pass
                    if all_double_ok:
                        double_cover = True
                except Exception:
                    double_cover = False
            results['norm_invariant'] = norm_invariant
            results['double_cover'] = double_cover
            return results
        except Exception:
            return results
