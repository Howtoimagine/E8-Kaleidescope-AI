"""E8 lattice physics and related verification/projection utilities."""
from __future__ import annotations
import os, math, numpy as np
from typing import Optional, Any, Dict
from collections import deque
try:
    from itertools import combinations
except Exception:
    combinations = None  # type: ignore
try:
    from scipy.spatial import KDTree  # type: ignore
except Exception:  # fallback minimal KDTree if scipy missing
    class KDTree:
        def __init__(self, data):
            self._data = np.asarray(data, dtype=float)
        def query(self, pts, k=1):
            Q = np.asarray(pts, dtype=float)
            dists = ((self._data[None, :, :] - Q[:, None, :])**2).sum(-1)**0.5
            idx = np.argmin(dists, axis=1)
            return dists[np.arange(len(Q)), idx], idx

# ---- Root System Generation & Verification ----

def generate_e8_roots():
    roots = set()
    if combinations is None: return np.array([])
    for i, j in combinations(range(8), 2):
        for s1, s2 in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            vec = [0]*8; vec[i], vec[j] = s1, s2
            roots.add(tuple(vec))
    for signs in range(2**8):
        vec, neg_count = [], 0
        for i in range(8):
            if (signs >> i) & 1: vec.append(-0.5); neg_count += 1
            else: vec.append(0.5)
        if neg_count % 2 == 0: roots.add(tuple(vec))
    return np.array(list(roots))

def verify_e8_roots(roots: np.ndarray, console=None) -> bool:
    try:
        if roots.size == 0:
            if console: console.log("[E8_VERIFY] No roots to verify")
            return False
        squared_norms = np.sum(roots * roots, axis=1)
        norm_check = np.allclose(squared_norms, 2.0, atol=1e-10)
        if not norm_check:
            if console: console.log("[E8_VERIFY] Norm check failed")
            return False
        dot_products = roots @ roots.T
        np.fill_diagonal(dot_products, 0)
        valid_dots = np.isin(np.round(dot_products), [-2, -1, 0, 1])
        if not np.all(valid_dots):
            if console: console.log("[E8_VERIFY] Inner product check failed")
            return False
        if console: console.log(f"✅ [E8_VERIFY] Verified {len(roots)} roots")
        return True
    except Exception as e:
        if console: console.log(f"[E8_VERIFY] Error: {e}")
        return False

def verify_e8_brackets(roots: np.ndarray, console=None) -> tuple[bool,int,int]:
    try:
        if roots.size == 0:
            return False,0,0
        n_roots = len(roots)
        bracket_matrix = np.zeros((n_roots, n_roots), dtype=int)
        for i in range(n_roots):
            for j in range(i+1, n_roots):
                alpha, beta = roots[i], roots[j]
                dot_product = np.dot(alpha, beta)
                if dot_product == 0:
                    bracket_matrix[i,j] = 0
                elif abs(dot_product) == 1:
                    gamma = (alpha + beta) / 2
                    gamma_scaled = 2 * gamma
                    distances = np.sum((roots - gamma_scaled)**2, axis=1)
                    bracket_matrix[i,j] = 1 if np.min(distances) < 1e-10 else 0
                else:
                    bracket_matrix[i,j] = -1
        bracket_matrix = bracket_matrix + bracket_matrix.T
        valid_brackets = np.sum(bracket_matrix == -1) == 0
        total_rel = n_roots * (n_roots - 1) // 2
        valid_rel = np.sum(bracket_matrix >= 0) // 2
        if console:
            console.log(f"🔀 [E8_BRACKET] {valid_rel}/{total_rel} valid")
        return bool(valid_brackets), int(valid_rel), int(total_rel)
    except Exception as e:
        if console: console.log(f"[E8_BRACKET] Error: {e}")
        return False,0,0

def triacontagonal_projection(roots: np.ndarray, console=None) -> np.ndarray:
    try:
        if roots.size == 0:
            return np.array([])
        phi = (1 + np.sqrt(5)) / 2
        projection_matrix = np.array([
            [1, phi, 0, -phi, -1, 0, phi, 1],
            [phi, 0, -phi, -1, 0, phi, 1, -phi],
            [0, 1, phi, 0, -1, -phi, 0, 1]
        ], dtype=float)
        projection_matrix = projection_matrix / np.linalg.norm(projection_matrix, axis=0)
        projected = roots @ projection_matrix.T
        if console:
            console.log(f"📐 [TRIACONTAGONAL] Projected {len(roots)} roots")
        return projected
    except Exception as e:
        if console: console.log(f"[TRIACONTAGONAL] Error: {e}")
        return np.array([])

def build_weighted_adjacency(roots, atol=1e-6):
    R = roots.astype(np.float32)
    N = R.shape[0]
    mask = np.isclose(np.abs(R @ R.T), 1.0, atol=atol)
    np.fill_diagonal(mask, False)
    W = np.zeros((N, N), dtype=np.float32); W[mask] = 1.0
    int_roots = {tuple((2*r).astype(np.int8)) for r in R}
    for i in range(N):
        for j in np.where(mask[i])[0]:
            ri2, rj2 = (2*R[i]).astype(np.int8), (2*R[j]).astype(np.int8)
            s, d = tuple((ri2 + rj2).tolist()), tuple((ri2 - rj2).tolist())
            W[i, j] += 0.15 * (s in int_roots) + 0.10 * (d in int_roots)
    return W

def all_pairs_hops(A_bool):
    N = A_bool.shape[0]
    nbrs = [np.where(A_bool[i] > 0)[0] for i in range(N)]
    dist = np.full((N, N), np.inf, dtype=np.float32)
    for s in range(N):
        dist[s, s] = 0.0; q = deque([s])
        while q:
            u = q.popleft()
            for v in nbrs[u]:
                if dist[s, v] == np.inf:
                    dist[s, v] = dist[s, u] + 1.0
                    q.append(v)
    return dist

class E8Physics:
    def __init__(self, console):
        self.console = console
        self.roots = generate_e8_roots()
        self.roots_unit = self.roots / (np.linalg.norm(self.roots, axis=1, keepdims=True) + 1e-12)
        self.roots_kdtree = KDTree(self.roots)
        self.weights = build_weighted_adjacency(self.roots)
        self.adj_bool = (self.weights > 0).astype(np.int8)
        self.hops = all_pairs_hops(self.adj_bool)
        # Laplacian + caches
        self.L_norm = self._build_normalized_laplacian()
        self._mask_cache = {}
        self.projection_matrix = None
        self._decoding_axis_idx = None
        self._decoding_axis_vec = None
        self._decoding_basis = None
        try:
            self.console.log(f"🔗 [INIT] E8Physics: roots={len(self.roots)}, edges={(self.adj_bool.sum())//2}")
        except Exception:
            pass
        # Run validation suite
        self._run_e8_validations()

    def find_nearest_root_index(self, vector_8d: np.ndarray) -> Optional[int]:
        if vector_8d is None or vector_8d.shape[0] != 8:
            return None
        try:
            _, index = self.roots_kdtree.query(vector_8d.reshape(1, -1), k=1)
            arr = np.atleast_1d(index).reshape(-1)
            if arr.size == 0:
                return None
            return int(arr[0])
        except Exception as e:
            try:
                self.console.log(f"[E8Physics] Error finding nearest root: {e}")
            except Exception:
                pass
            return None

    def get_symmetric_counterpart(self, index: int) -> int:
        try:
            if not hasattr(self, 'roots') or index < 0 or index >= len(self.roots):
                return -1
            phi = (1 + np.sqrt(5)) / 2
            base_vec = self.roots[index]
            target_vector = -base_vec * phi
            _, idx_arr = self.roots_kdtree.query(target_vector.reshape(1, -1), k=1)
            counterpart = int(np.atleast_1d(idx_arr).reshape(-1)[0])
            if counterpart == index:
                target_vector2 = -base_vec
                _, idx_arr2 = self.roots_kdtree.query(target_vector2.reshape(1, -1), k=1)
                counterpart = int(np.atleast_1d(idx_arr2).reshape(-1)[0])
            return counterpart if 0 <= counterpart < len(self.roots) else -1
        except Exception:
            return -1

    # --- Added methods ported from monolith ---
    def _build_normalized_laplacian(self):
        """Return normalized graph Laplacian (I - D^{-1/2} W D^{-1/2}). Fallback: identity if scipy sparse missing."""
        try:
            from scipy.sparse import csr_matrix  # type: ignore
            from scipy.sparse import diags       # separated for static analyzer
        except Exception:
            return np.eye(self.weights.shape[0], dtype=np.float32)
        try:
            W = csr_matrix(self.weights, dtype=np.float32)
            n = int(getattr(W, 'shape', (0,0))[0])
            if n <= 0:
                return np.eye(self.weights.shape[0], dtype=np.float32)
            deg = np.asarray(W.sum(axis=1)).ravel()
            inv = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
            D_inv_sqrt = diags(inv)
            I = diags(np.ones(n, dtype=np.float32))
            Lmat = (I - D_inv_sqrt @ W @ D_inv_sqrt).astype(np.float32)
            return Lmat
        except Exception:
            return np.eye(self.weights.shape[0], dtype=np.float32)

    def heat_mask_cached(self, center_idx: int, sigma: float = 1.25):
        """Gaussian heat mask over hop distances, cached by (index,sigma)."""
        key = (int(center_idx), round(float(sigma), 2))
        m = self._mask_cache.get(key)
        if m is None:
            try:
                d = self.hops[center_idx]
                m = np.exp(- (d * d) / (2.0 * sigma * sigma)).astype(np.float32)
            except Exception:
                m = np.ones(len(self.roots), dtype=np.float32)
            self._mask_cache[key] = m
        return m

    def _run_e8_validations(self):
        """Run E8 root/bracket/structure validations (errors non-fatal)."""
        try:
            roots_valid = verify_e8_roots(self.roots, self.console)
            # Brackets
            _br_ok, _v, _t = normalize_bracket_result(verify_e8_brackets(self.roots, self.console))
            brackets_valid = (_t > 0 and _v == _t)
            # Lie algebra test (placeholder minimal until full extraction)
            try:
                lie_valid = verify_e8_lie_algebra_structure(self.roots, self.console)  # type: ignore
            except Exception:
                lie_valid = False
            # Octonion structural test (optional)
            try:
                octonion_valid = octonion_bracket_test(self.console)  # type: ignore
            except Exception:
                octonion_valid = False
            # Projection sanity
            try:
                triacontagonal_projection(self.roots, self.console)
            except Exception:
                pass
            all_valid = roots_valid and brackets_valid and lie_valid and octonion_valid
            status = "✅" if all_valid else "⚠️"
            if self.console:
                self.console.log(f"🎯 [E8_VERIFY] {status} roots={roots_valid} brackets={brackets_valid} lie={lie_valid} octonion={octonion_valid}")
        except Exception as e:
            if self.console:
                self.console.log(f"[E8_VERIFY] validation error: {e}")

    # Decoding axis helpers
    def get_decoding_axis_unit(self) -> np.ndarray:
        try:
            idx_env = os.getenv("E8_DECODING_ROOT_INDEX")
            idx = int(idx_env) % len(self.roots_unit) if idx_env else 0
        except Exception:
            idx = 0
        if self._decoding_axis_idx != idx or self._decoding_axis_vec is None:
            try:
                v = np.asarray(self.roots_unit[idx], dtype=np.float32).copy()
            except Exception:
                v = np.ones(8, dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                v = np.ones(8, dtype=np.float32) / np.sqrt(8.0)
            self._decoding_axis_vec = v
            self._decoding_axis_idx = idx
        return self._decoding_axis_vec

    def get_decoding_basis(self) -> np.ndarray:
        if self._decoding_basis is not None:
            return self._decoding_basis
        try:
            v = np.asarray(self.get_decoding_axis_unit(), dtype=np.float32).reshape(-1)
            if v.shape[0] != 8:
                v = np.ones(8, dtype=np.float32) / np.sqrt(8.0)
            v = v / (np.linalg.norm(v) + 1e-12)
            e1 = np.zeros(8, dtype=np.float32); e1[0] = 1.0
            diff = v - e1
            norm_diff = float(np.linalg.norm(diff))
            if norm_diff < 1e-9:
                H = np.eye(8, dtype=np.float32)
            else:
                u = diff / norm_diff
                H = np.eye(8, dtype=np.float32) - 2.0 * np.outer(u, u)
                try:
                    Qh, _ = np.linalg.qr(H)
                    H = Qh.astype(np.float32)
                except Exception:
                    pass
            self._decoding_basis = H
            return self._decoding_basis
        except Exception as e:
            if self.console:
                try: self.console.log(f"[E8Physics] get_decoding_basis failed: {e}")
                except Exception: pass
            return np.eye(8, dtype=np.float32)

    # Blueprint generation (quasicrystal style random orthogonal 3D proj)
    def generate_quasicrystal_blueprint(self, seed: int = 0):
        P, pts = None, None
        uniqueness_threshold = 230
        max_tries = 32
        for i in range(max_tries):
            current_seed = seed + i
            rng = np.random.default_rng(current_seed)
            M = rng.normal(size=(8, 3)).astype(np.float32)
            Q, _ = np.linalg.qr(M)
            P_candidate = Q[:, :3]
            pts_candidate = self.roots @ P_candidate
            unique_pts = np.unique(np.round(pts_candidate, 3), axis=0)
            if len(unique_pts) >= uniqueness_threshold:
                P = P_candidate; pts = pts_candidate
                if self.console:
                    self.console.log(f"🔮 [INIT] Quasicrystal projection found after {i+1} tries. Uniqueness: {len(unique_pts)}/240.")
                break
        if P is None:
            rng = np.random.default_rng(seed + max_tries - 1)
            M = rng.normal(size=(8, 3)).astype(np.float32)
            Q, _ = np.linalg.qr(M)
            P = Q[:, :3]; pts = self.roots @ P
            if self.console:
                self.console.log("[WARN] Quasicrystal projection fallback using last attempt.")
        if pts is None or getattr(pts, 'size', 0) == 0:
            return []
        pts = pts - pts.mean(axis=0, keepdims=True)
        pts = pts / (np.abs(pts).max() + 1e-6)
        self.projection_matrix = P
        blueprint_coords: list[dict[str, float]] = []
        rounded_coords = np.round(pts, 4)
        from collections import defaultdict as _dd
        coord_groups: Dict[tuple, list[int]] = _dd(list)  # correct value type
        for i, coord in enumerate(rounded_coords):
            coord_groups[tuple(coord)].append(int(i))
        for i in range(pts.shape[0]):
            base_x, base_y, base_z = float(pts[i,0]), float(pts[i,1]), float(pts[i,2])
            group = coord_groups[tuple(rounded_coords[i])]
            render_x, render_y = base_x, base_y
            if len(group) > 1:
                k = group.index(i) + 1
                epsilon = 0.005
                radius = epsilon * math.sqrt(k)
                theta = k * math.pi * (3 - math.sqrt(5))
                render_x += radius * math.cos(theta)
                render_y += radius * math.sin(theta)
            blueprint_coords.append({
                "id": i, "x": base_x, "y": base_y, "z": base_z,
                "render_x": render_x, "render_y": render_y, "render_z": base_z
            })
        # min neighbor distance diagnostic
        try:
            kdtree = KDTree(pts)
            distances, _ = kdtree.query(pts, k=2)
            min_dist = float(np.min(distances[:,1]))
            if self.console:
                self.console.log(f"📏 [INIT] Min nearest-neighbor distance in blueprint: {min_dist:.4f}")
        except Exception:
            pass
        return blueprint_coords

# Helper moved from monolith
def normalize_bracket_result(res) -> tuple:
    try:
        if isinstance(res, (np.bool_, bool)):
            return bool(res), 0, 0
        if isinstance(res, (tuple, list)):
            if len(res) == 3:
                b, v, t = res; return bool(b), int(v), int(t)
            if len(res) == 2:
                v, t = res; b = (int(t) > 0 and int(v) == int(t)); return bool(b), int(v), int(t)
        return False, 0, 0
    except Exception:
        return False, 0, 0

# Placeholders for not-yet-extracted advanced validation routines
def verify_e8_lie_algebra_structure(roots: np.ndarray, console=None):  # type: ignore
    if console:
        console.log("[E8_LIE] Placeholder lie algebra structure test (always False until extracted)")
    return False

def octonion_bracket_test(console=None):  # type: ignore
    if console:
        console.log("[OCTONION] Placeholder octonion bracket test (False)")
    return False
