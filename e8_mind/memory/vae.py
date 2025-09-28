from __future__ import annotations

from typing import Any, Optional
import os
import numpy as np


class _StubProjector:
    """Simple projection stub that pads/trims and L2-normalizes.

    Used when torch or a learned VAE isn't available.
    """

    def __init__(self):
        self.is_trained = False

    def project_to_dim(self, v: np.ndarray, d: int = 8) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.size != d:
            out = np.zeros(d, dtype=np.float32)
            n = min(arr.size, d)
            out[:n] = arr[:n]
            arr = out
        nrm = float(np.linalg.norm(arr))
        if nrm > 1e-6:
            arr = arr / nrm
        return arr.astype(np.float32)


class VariationalAutoencoder:
    """Optional VAE wrapper.

    If PyTorch is available, this class can be extended to load a model and
    run inference. For now, we provide a minimal interface that mirrors the
    legacy usage: a .project_to_dim method and an is_trained flag.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model = None
        self.device = device or os.getenv("E8_TORCH_DEVICE", "cpu")
        self.is_trained = False
        try:
            import torch  # type: ignore
            self.torch = torch  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - optional dependency
            self.torch = None

        if self.torch is not None and model_path:
            self._try_load_torch_model(model_path)
        if self.model is None:
            # Fallback to stub
            self.model = _StubProjector()
            self.is_trained = False

    def _try_load_torch_model(self, model_path: str):  # pragma: no cover - best effort
        try:
            if self.torch is None:
                return
            # Placeholder: expects a scripted module with forward(x) -> latent or projected
            m = self.torch.jit.load(model_path, map_location=self.device)  # type: ignore[attr-defined]
            m.eval()
            self.model = m
            self.is_trained = True
        except Exception:
            self.model = None
            self.is_trained = False

    def project_to_dim(self, v: np.ndarray, d: int = 8):
        if self.torch is not None and self.is_trained and self.model is not None:
            try:
                with self.torch.inference_mode():
                    x = self.torch.as_tensor(np.asarray(v, dtype=np.float32)).to(self.device).reshape(1, -1)
                    z = self.model(x)  # type: ignore[misc]
                    if isinstance(z, (tuple, list)):
                        z = z[0]
                    z = z.reshape(-1)
                    # If output dim doesn't match, pad/trim deterministically
                    out = z
                    if z.numel() != int(d):
                        pad = self.torch.zeros(int(d), device=z.device, dtype=z.dtype)
                        n = min(z.numel(), int(d))
                        pad[:n] = z[:n]
                        out = pad
                    nrm = self.torch.linalg.norm(out)
                    out = out / (nrm + 1e-12)
                    return out
            except Exception:
                pass
        # Fallback path
        return _StubProjector().project_to_dim(v, d)
