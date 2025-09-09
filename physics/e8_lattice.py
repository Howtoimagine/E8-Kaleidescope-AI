"""
E8 Lattice Physics Engine

This module contains the core E8 physics implementation including root generation,
adjacency matrices, and lattice operations.
"""

import os
import math
import time
import numpy as np
from collections import deque, defaultdict
from typing import Optional, List, Any, Dict, Tuple

try:
    from itertools import combinations
except Exception:
    combinations = None

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.spatial import KDTree as _SPKDTree
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh, expm_multiply
    import scipy as sp
except Exception:
    _SPKDTree, csr_matrix, diags, eigsh, expm_multiply, sp = None, None, None, None, None, None

from core.data_structures import KDTree


def generate_e8_roots():
    """Generate the 240 roots of the E8 lattice."""
    roots = set()
    if combinations is None:
        return np.array([])
    
    # All combinations of 2 coordinates being ±1, others 0
    for i, j in combinations(range(8), 2):
        for s1, s2 in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            vec = [0]*8
            vec[i], vec[j] = s1, s2
            roots.add(tuple(vec))
    
    # All combinations where coordinates are ±0.5, with even number of minus signs
    for signs in range(2**8):
        vec, neg_count = [], 0
        for i in range(8):
            if (signs >> i) & 1:
                vec.append(-0.5)
                neg_count += 1
            else:
                vec.append(0.5)
        if neg_count % 2 == 0:
            roots.add(tuple(vec))
    
    return np.array(list(roots))


def build_weighted_adjacency(roots, atol=1e-6):
    """Build weighted adjacency matrix for E8 roots."""
    R = roots.astype(np.float32)
    N = R.shape[0]
    
    # Basic adjacency: roots with dot product ±1
    mask = np.isclose(np.abs(R @ R.T), 1.0, atol=atol)
    np.fill_diagonal(mask, False)
    
    W = np.zeros((N, N), dtype=np.float32)
    W[mask] = 1.0
    
    # Enhanced weights based on lattice structure
    int_roots = {tuple((2*r).astype(np.int8)) for r in R}
    for i in range(N):
        for j in np.where(mask[i])[0]:
            ri2, rj2 = (2*R[i]).astype(np.int8), (2*R[j]).astype(np.int8)
            s, d = tuple((ri2 + rj2).tolist()), tuple((ri2 - rj2).tolist())
            W[i, j] += 0.15 * (s in int_roots) + 0.10 * (d in int_roots)
    
    return W


def build_diff_adjacency(roots):
    """Build difference-based adjacency matrix."""
    R = roots.astype(np.float32)
    N = R.shape[0]
    int_roots = set(tuple((2*r).astype(np.int8)) for r in R)
    
    Wd = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        ri2 = tuple((2*R[i]).astype(np.int8))
        for j in range(N):
            if i == j:
                continue
            rj2 = tuple((2*R[j]).astype(np.int8))
            d = tuple(a - b for a, b in zip(ri2, rj2))
            if d in int_roots:
                Wd[i, j] = 1.0
    
    # Make symmetric
    Wd = 0.5 * (Wd + Wd.T)
    np.fill_diagonal(Wd, 0.0)
    return Wd


def all_pairs_hops(A_bool):
    """Compute all-pairs shortest path distances."""
    N = A_bool.shape[0]
    nbrs = [np.where(A_bool[i] > 0)[0] for i in range(N)]
    
    dist = np.full((N, N), np.inf, dtype=np.float32)
    
    for s in range(N):
        dist[s, s] = 0.0
        q = deque([s])
        
        while q:
            u = q.popleft()
            for v in nbrs[u]:
                if dist[s, v] == np.inf:
                    dist[s, v] = dist[s, u] + 1.0
                    q.append(v)
    
    return dist


def weyl_average_potential(physics, anchors, draws=3, seed=None):
    """Apply Weyl group transformations to average potentials."""
    rng = np.random.default_rng(seed)
    V_acc = np.zeros(physics.weights.shape[0], dtype=np.float32)
    
    def rand_sign_perm(rng):
        P = np.eye(8, dtype=np.float32)
        rng.shuffle(P)
        signs = rng.choice([-1.0, 1.0], size=(8,), replace=True).astype(np.float32)
        if (signs < 0).sum() % 2 == 1:
            signs[0] *= -1.0
        return (P.T * signs).T
    
    for _ in range(draws):
        A = rand_sign_perm(rng)
        transformed = []
        for (s, lam) in anchors.anchors:
            sA = (A @ s).astype(np.float32)
            sA /= np.linalg.norm(sA) + 1e-12
            transformed.append((sA, lam))
        
        # Create temporary anchor field
        from .potential_fields import MultiAnchorField
        tmp = MultiAnchorField(physics, kernel=anchors.kernel, rbf_sigma=anchors.rbf_sigma)
        tmp.set(transformed)
        V_acc += tmp.potential()
    
    return (V_acc / float(draws)).astype(np.float32)


def add_curiosity_penalty(V, visits, alpha=0.12):
    """Add curiosity-driven penalty based on visit counts."""
    try:
        cur = -alpha * np.log1p(visits.astype(np.float32))
        return (V + cur).astype(np.float32)
    except Exception:
        return V


class E8Physics:
    """
    Core physics engine for E8 lattice operations.
    
    Manages the 240 E8 roots, their adjacency relationships,
    and provides geometric operations in the E8 space.
    """
    
    def __init__(self, console):
        self.console = console
        
        # Generate E8 roots and derived structures
        self.roots = generate_e8_roots()
        self.roots_unit = self.roots / (np.linalg.norm(self.roots, axis=1, keepdims=True) + 1e-12)
        self.roots_kdtree = KDTree(self.roots)
        
        # Build adjacency matrices
        self.weights = build_weighted_adjacency(self.roots)
        self.adj_bool = (self.weights > 0).astype(np.int8)
        self.hops = all_pairs_hops(self.adj_bool)
        
        # Build normalized Laplacian
        self.L_norm = self._build_normalized_laplacian()
        
        # Cache for heat masks
        self._mask_cache = {}
        
        # Projection matrix for quasicrystal blueprint
        self.projection_matrix = None
        
        self.console.log(f"[INIT] E8Physics: roots={len(self.roots)}, edges={(self.adj_bool.sum())//2}")

    def find_nearest_root_index(self, vector_8d: np.ndarray) -> Optional[int]:
        """Find the index of the nearest E8 root to a given 8D vector."""
        if vector_8d is None or vector_8d.shape[0] != 8:
            return None
        
        try:
            _, index = self.roots_kdtree.query(vector_8d.reshape(1, -1), k=1)
            result_index = index[0] if isinstance(index, np.ndarray) else index
            return int(result_index)
        except Exception as e:
            self.console.log(f"[E8Physics] Error finding nearest root: {e}")
            return None

    def generate_quasicrystal_blueprint(self, seed: Optional[int] = None):
        """Generate a 3D quasicrystal projection of the E8 roots."""
        if seed is None:
            seed = int(os.getenv("GLOBAL_SEED", "42"))
        
        P, pts = None, None
        uniqueness_threshold = 230
        max_tries = 32

        for i in range(max_tries):
            current_seed = seed + i
            rng = np.random.default_rng(current_seed)
            
            # Generate random 8x3 matrix and orthogonalize
            M = rng.normal(size=(8, 3)).astype(np.float32)
            Q, _ = np.linalg.qr(M)
            P_candidate = Q[:, :3]

            # Project E8 roots to 3D
            pts_candidate = self.roots @ P_candidate
            unique_pts = np.unique(np.round(pts_candidate, 3), axis=0)

            if len(unique_pts) >= uniqueness_threshold:
                P = P_candidate
                pts = pts_candidate
                self.console.log(f"[INIT] Quasicrystal projection found after {i+1} tries. Uniqueness: {len(unique_pts)}/240.")
                break

        if P is None:
            self.console.log(f"[bold yellow][WARN] Quasicrystal projection failed to meet uniqueness threshold after {max_tries} tries. Using last attempt.[/bold yellow]")
            rng = np.random.default_rng(seed + max_tries - 1)
            M = rng.normal(size=(8, 3)).astype(np.float32)
            Q, _ = np.linalg.qr(M)
            P = Q[:, :3]
            pts = self.roots @ P

        # Normalize the projection
        assert pts is not None, "Projection points (pts) must be computed before normalization."
        pts -= pts.mean(axis=0, keepdims=True)
        pts /= (np.abs(pts).max() + 1e-6)
        self.projection_matrix = P

        # Create blueprint coordinates with collision resolution
        blueprint_coords = []
        rounded_coords = np.round(pts, 4)
        coord_groups = defaultdict(list)
        
        for i, coord in enumerate(rounded_coords):
            coord_groups[tuple(coord)].append(i)

        for i in range(pts.shape[0]):
            base_x, base_y, base_z = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
            group = coord_groups[tuple(rounded_coords[i])]
            render_x, render_y = base_x, base_y

            # Resolve collisions with spiral pattern
            if len(group) > 1:
                k = group.index(i) + 1
                epsilon = 0.005
                radius = epsilon * math.sqrt(k)
                theta = k * math.pi * (3 - math.sqrt(5))  # Golden angle
                render_x += radius * math.cos(theta)
                render_y += radius * math.sin(theta)

            blueprint_coords.append({
                "id": i,
                "x": base_x, "y": base_y, "z": base_z,
                "render_x": render_x, "render_y": render_y, "render_z": base_z
            })

        # Calculate minimum distance for debugging
        try:
            kdtree = KDTree(pts)
            distances, _ = kdtree.query(pts, k=2)
            min_dist = np.min(distances[:, 1])
            self.console.log(f"[INIT] Min nearest-neighbor distance in blueprint: {min_dist:.4f}")
        except Exception as e:
            self.console.log(f"[INIT] Could not calculate min distance: {e}")

        return blueprint_coords

    def _build_normalized_laplacian(self):
        """Build the normalized Laplacian matrix."""
        if csr_matrix is None or diags is None:
            # Pure numpy fallback
            W = np.asarray(self.weights, dtype=np.float32)
            deg = np.sum(W, axis=1)
            d_is = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
            D_inv_sqrt = np.diag(d_is)
            return np.eye(W.shape[0], dtype=np.float32) - D_inv_sqrt @ W @ D_inv_sqrt
        
        # Sparse matrix version
        W = csr_matrix(self.weights, dtype=np.float32)
        deg = np.asarray(W.sum(axis=1)).ravel()
        D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
        n = int(np.shape(self.weights)[0])
        return diags(np.ones(n, dtype=np.float32)) - D_inv_sqrt @ W @ D_inv_sqrt

    def heat_mask_cached(self, center_idx, sigma=1.25):
        """Get cached heat diffusion mask around a center point."""
        key = (int(center_idx), round(float(sigma), 2))
        m = self._mask_cache.get(key)
        
        if m is None:
            d = self.hops[center_idx]
            m = np.exp(- (d * d) / (2.0 * sigma * sigma)).astype(np.float32)
            self._mask_cache[key] = m
        
        return m
