from __future__ import annotations
from typing import Dict, Tuple, Optional, Any
import numpy as np
import logging

# Optional torch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False

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

__all__ = ["SubspaceProjector", "VariationalAutoencoder"]


class VariationalAutoencoder:
    """
    Lightweight fallback VAE with shape-only projections.
    
    Provides either full PyTorch VAE implementation when available,
    or simple projection-based fallback when torch is unavailable.
    Methods are compatible for environments without the full VAE impl.
    """
    
    def __init__(self, layer_sizes: Optional[list] = None, console: Any = None):
        self._trained = False
        self.layer_sizes = layer_sizes or []
        self.console = console
        self.device = None
        
        # Initialize torch components if available
        if TORCH_AVAILABLE and torch is not None:
            self._init_torch_vae()
        else:
            self._init_fallback_vae()

    def _init_torch_vae(self):
        """Initialize full PyTorch VAE implementation."""
        if not self.layer_sizes or len(self.layer_sizes) < 2:
            # Default architecture
            self.layer_sizes = [768, 256, 128, 64]
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
        
        input_dim = self.layer_sizes[0]
        hidden_dims = self.layer_sizes[1:-1]
        latent_dim = self.layer_sizes[-1]
        
        # Encoder
        encoder_layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),  # type: ignore
                nn.ReLU(),  # type: ignore
                nn.Dropout(0.1)  # type: ignore
            ])
            current_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)  # type: ignore
        
        # Latent space
        self.fc_mu = nn.Linear(current_dim, latent_dim)  # type: ignore
        self.fc_logvar = nn.Linear(current_dim, latent_dim)  # type: ignore
        
        # Decoder
        decoder_layers = []
        current_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),  # type: ignore
                nn.ReLU(),  # type: ignore
                nn.Dropout(0.1)  # type: ignore
            ])
            current_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(current_dim, input_dim))  # type: ignore
        self.decoder = nn.Sequential(*decoder_layers)  # type: ignore
        
        # Move to device
        self.encoder = self.encoder.to(self.device)  # type: ignore
        self.fc_mu = self.fc_mu.to(self.device)  # type: ignore
        self.fc_logvar = self.fc_logvar.to(self.device)  # type: ignore
        self.decoder = self.decoder.to(self.device)  # type: ignore
        
        # Optimizer
        all_params = (list(self.encoder.parameters()) +  # type: ignore
                     list(self.fc_mu.parameters()) +  # type: ignore
                     list(self.fc_logvar.parameters()) +  # type: ignore
                     list(self.decoder.parameters()))  # type: ignore
        self.optimizer = torch.optim.Adam(all_params, lr=1e-3)  # type: ignore
        
        self.torch_available = True

    def _init_fallback_vae(self):
        """Initialize simple projection-based fallback."""
        self.torch_available = False
        if not self.layer_sizes:
            self.layer_sizes = [768, 64]  # Simple input -> latent mapping
            
        # Use SubspaceProjector for deterministic projections
        self.projector = SubspaceProjector(seed=42)

    @property
    def is_trained(self) -> bool:
        return bool(self._trained)

    def train_on_batch(self, x: np.ndarray) -> Dict[str, float]:
        """
        Train on a batch of data.
        
        Args:
            x: Input data (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Dictionary with loss components
        """
        if self.torch_available and torch is not None:
            return self._train_torch_batch(x)
        else:
            return self._train_fallback_batch(x)
    
    def _train_torch_batch(self, x: np.ndarray) -> Dict[str, float]:
        """Train using PyTorch VAE."""
        try:
            # Convert to tensor
            x_tensor = torch.FloatTensor(x).to(self.device)  # type: ignore
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(0)
                
            self.optimizer.zero_grad()  # type: ignore
            
            # Forward pass
            encoded = self.encoder(x_tensor)  # type: ignore
            mu = self.fc_mu(encoded)  # type: ignore
            logvar = self.fc_logvar(encoded)  # type: ignore
            
            # Reparameterization
            z = self._reparameterize(mu, logvar)
            
            # Decode
            x_recon = self.decoder(z)  # type: ignore
            
            # Compute losses
            recon_loss = F.mse_loss(x_recon, x_tensor, reduction='mean')  # type: ignore
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_tensor.size(0)  # type: ignore
            total_loss = recon_loss + 0.1 * kld_loss  # Beta-VAE with beta=0.1
            
            # Backward pass
            total_loss.backward()  # type: ignore
            self.optimizer.step()  # type: ignore
            
            self._trained = True
            
            return {
                "total_loss": float(total_loss.item()),  # type: ignore
                "recon_loss": float(recon_loss.item()),  # type: ignore
                "kld_loss": float(kld_loss.item())  # type: ignore
            }
            
        except Exception as e:
            if self.console:
                self.console.log(f"[red]VAE training error: {e}[/red]")
            return {"total_loss": 0.0, "recon_loss": 0.0, "kld_loss": 0.0}
    
    def _train_fallback_batch(self, x: np.ndarray) -> Dict[str, float]:
        """Fallback training (no-op for projector)."""
        self._trained = True
        return {"total_loss": 0.0, "recon_loss": 0.0, "kld_loss": 0.0}
    
    def _reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        if self.torch_available and torch is not None:
            std = torch.exp(0.5 * logvar)  # type: ignore
            eps = torch.randn_like(std)  # type: ignore
            return mu + eps * std
        else:
            return mu  # Fallback: just return mean

    def _to_np(self, x: Any) -> Optional[np.ndarray]:
        """Convert various input types to numpy array."""
        try:
            if x is None:
                return None
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            return np.asarray(x, dtype=np.float32)
        except Exception:
            return x

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input to latent space.
        
        Args:
            x: Input data (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Latent representation
        """
        if self.torch_available and torch is not None and self.is_trained:
            return self._encode_torch(x)
        else:
            return self._encode_fallback(x)
    
    def _encode_torch(self, x: np.ndarray) -> np.ndarray:
        """Encode using PyTorch VAE."""
        try:
            with torch.no_grad():  # type: ignore
                x_tensor = torch.FloatTensor(x).to(self.device)  # type: ignore
                if x_tensor.dim() == 1:
                    x_tensor = x_tensor.unsqueeze(0)
                    
                encoded = self.encoder(x_tensor)  # type: ignore
                mu = self.fc_mu(encoded)  # type: ignore
                result = self._to_np(mu)
                return result if result is not None else self._encode_fallback(x)
        except Exception:
            return self._encode_fallback(x)
    
    def _encode_fallback(self, x: np.ndarray) -> np.ndarray:
        """Encode using projection fallback."""
        if not self.layer_sizes:
            return x
        target_dim = self.layer_sizes[-1]
        return self.projector.project_to_dim(x, target_dim)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent representation back to input space.
        
        Args:
            z: Latent representation
            
        Returns:
            Reconstructed input
        """
        if self.torch_available and torch is not None and self.is_trained:
            return self._decode_torch(z)
        else:
            return self._decode_fallback(z)
    
    def _decode_torch(self, z: np.ndarray) -> np.ndarray:
        """Decode using PyTorch VAE."""
        try:
            with torch.no_grad():  # type: ignore
                z_tensor = torch.FloatTensor(z).to(self.device)  # type: ignore
                if z_tensor.dim() == 1:
                    z_tensor = z_tensor.unsqueeze(0)
                    
                x_recon = self.decoder(z_tensor)  # type: ignore
                result = self._to_np(x_recon)
                return result if result is not None else self._decode_fallback(z)
        except Exception:
            return self._decode_fallback(z)
    
    def _decode_fallback(self, z: np.ndarray) -> np.ndarray:
        """Decode using projection fallback (expand with zeros)."""
        if not self.layer_sizes:
            return z
        input_dim = self.layer_sizes[0]
        z_np = np.asarray(z, dtype=np.float32)
        if z_np.ndim == 1:
            result = np.zeros(input_dim, dtype=np.float32)
            result[:min(len(z_np), input_dim)] = z_np[:min(len(z_np), input_dim)]
            return result
        else:
            # Batch processing
            batch_size = z_np.shape[0]
            result = np.zeros((batch_size, input_dim), dtype=np.float32)
            min_dim = min(z_np.shape[1], input_dim)
            result[:, :min_dim] = z_np[:, :min_dim]
            return result

    def project_to_dim(self, x: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Project input to target dimension.
        Maintains interface compatibility with existing code.
        
        Args:
            x: Input data
            target_dim: Target dimension
            
        Returns:
            Projected data
        """
        x_np = self._to_np(x)
        if x_np is None:
            return np.zeros(target_dim, dtype=np.float32)
            
        x2 = np.atleast_2d(x_np).astype(np.float32)
        out = np.zeros((x2.shape[0], int(target_dim)), dtype=np.float32)
        n = min(x2.shape[1], int(target_dim))
        out[:, :n] = x2[:, :n]
        return out if x2.ndim == 2 else out[0]

    def project_between_dim(self, x: np.ndarray, source_dim: int, target_dim: int) -> np.ndarray:
        """
        Project between dimensions.
        Ignores source_dim and just reshapes as needed.
        
        Args:
            x: Input data
            source_dim: Source dimension (ignored)
            target_dim: Target dimension
            
        Returns:
            Projected data
        """
        return self.project_to_dim(x, target_dim)

    def get_latent_dim(self) -> int:
        """Get the latent dimension."""
        if self.layer_sizes:
            return self.layer_sizes[-1]
        return 64  # Default latent dim
