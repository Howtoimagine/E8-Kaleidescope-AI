from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

class SubspaceProjector:
    """
    Lightweight, deterministic subspace projector.
    - Provides stable projection to target_dim via seeded orthonormal random basis.
    - No training required; `fit` is a no-op placeholder for future learned models.
    """

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._bases: Dict[Tuple[int, int], np.ndarray] = {}  # (input_dim, target_dim) -> (target_dim, input_dim)

    def fit(self, X: np.ndarray, target_dim: int) -> None:
        """
        Optional placeholder for future learned projectors (e.g., PCA/autoencoder).
        Currently a no-op: projections are deterministic from seed.
        """
        _ = (X, target_dim)  # silence linters

    def project_to_dim(self, x: np.ndarray, target_dim: int, normalize: bool = True) -> np.ndarray:
        """
        Project a single vector to `target_dim`.
        """
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        basis = self._get_basis(input_dim=v.shape[0], target_dim=int(target_dim))
        y = basis @ v  # (target_dim,)
        if normalize:
            n = float(np.linalg.norm(y))
            if n > 0:
                y = (y / n).astype(np.float32)
            else:
                y = y.astype(np.float32)
        else:
            y = y.astype(np.float32)
        return y

    def project_batch(self, X: np.ndarray, target_dim: int, normalize: bool = True) -> np.ndarray:
        """
        Project a batch (n, d_in) to (n, target_dim).
        """
        M = np.asarray(X, dtype=np.float32)
        if M.ndim == 1:
            return self.project_to_dim(M, target_dim=target_dim, normalize=normalize).reshape(1, -1)

        basis = self._get_basis(input_dim=M.shape[1], target_dim=int(target_dim))
        Y = (M @ basis.T).astype(np.float32)  # (n, target_dim)
        if normalize:
            norms = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
            Y = (Y / norms).astype(np.float32)
        return Y

    def _get_basis(self, input_dim: int, target_dim: int) -> np.ndarray:
        """
        Return a (target_dim, input_dim) orthonormal basis (rows) generated deterministically from seed.
        Cached per (input_dim, target_dim).
        """
        key = (int(input_dim), int(target_dim))
        B = self._bases.get(key)
        if B is not None:
            return B

        if target_dim <= 0 or input_dim <= 0:
            raise ValueError(f"Invalid dims: input_dim={input_dim}, target_dim={target_dim}")
        if target_dim > input_dim:
            raise ValueError(f"target_dim ({target_dim}) cannot exceed input_dim ({input_dim}).")

        rng = np.random.default_rng(self.seed + input_dim * 1009 + target_dim * 9176)
        M = rng.standard_normal((input_dim, target_dim)).astype(np.float32)  # (d_in, d_out)
        # Orthonormalize columns to get an orthonormal basis in the subspace
        Q, _ = np.linalg.qr(M)  # (d_in, d_out)
        B = Q.T.astype(np.float32)  # (d_out, d_in) so that y = B @ x
        self._bases[key] = B
        return B

__all__ = ["SubspaceProjector"]
