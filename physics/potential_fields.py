"""
Potential field implementations for E8 physics simulations.
"""

import numpy as np
from typing import List, Tuple

class MultiAnchorField:
    """
    Multi-anchor potential field for physics simulations.
    """
    def __init__(self, physics, kernel: str = 'cosine', rbf_sigma: float = 0.8):
        self.physics = physics
        self.kernel = kernel
        self.rbf_sigma = rbf_sigma
        self.anchors: List[Tuple[np.ndarray, float]] = []

    def set(self, anchor_list: List[Tuple[np.ndarray, float]]):
        """Set the anchors with their weights."""
        self.anchors = []
        if not anchor_list:
            return
        
        total_weight = sum(w for _, w in anchor_list)
        if total_weight > 1e-9:
            self.anchors = [(vec, w / total_weight) for vec, w in anchor_list]

    def potential(self) -> np.ndarray:
        """Calculate the potential field across all physics roots."""
        V = np.zeros(240, dtype=np.float32)
        if not self.anchors:
            return V
        
        for vec, weight in self.anchors:
            if self.kernel == 'cosine':
                scores = self.physics.roots_unit @ vec
            else:
                dists = np.linalg.norm(self.physics.roots - vec, axis=1)
                scores = np.exp(-dists**2 / (2 * self.rbf_sigma**2))
            V -= weight * scores
        
        return V
