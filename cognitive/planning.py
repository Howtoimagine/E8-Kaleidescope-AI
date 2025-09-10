"""
Planning and Decision-Making Components

This module contains cognitive architectures for planning, decision-making,
and adaptive parameter selection including contextual bandits and planners.
"""

from __future__ import annotations
import os
import random
import json
import numpy as np
from typing import Any, List, Dict, Optional, cast

# Safe JSON utilities - fallback implementations
def safe_json_read(path: str) -> Optional[Dict]:
    """Safely read JSON file with error handling."""
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return None

def safe_json_write(path: str, data: Dict) -> bool:
    """Safely write JSON file with error handling."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


class ContextBandit:
    """LinUCB contextual bandit for adaptive parameter selection."""
    
    def __init__(self, arms: List[Dict], state_dim: int, path_json: str, alpha: float = 1.0):
        self.arms = arms
        self.num_arms = len(arms)
        self.state_dim = state_dim
        self.path = path_json
        self.alpha = alpha
        self.A = [np.identity(state_dim, dtype=np.float64) for _ in range(self.num_arms)]
        self.b = [np.zeros((state_dim, 1), dtype=np.float64) for _ in range(self.num_arms)]
        self.load()

    def load(self):
        """Load learned bandit models from disk."""
        data = safe_json_read(self.path)
        if data and 'A' in data and 'b' in data:
            try:
                self.A = [np.array(a, dtype=np.float64) for a in data['A']]
                self.b = [np.array(bv, dtype=np.float64) for bv in data['b']]
                print("📈 [ContextBandit] Loaded learned models.")
            except Exception as e:
                print(f"[ContextBandit] Failed to load models, resetting: {e}")

    def save(self):
        """Save learned bandit models to disk."""
        try:
            safe_json_write(self.path, {
                'A': [a.tolist() for a in self.A],
                'b': [bv.tolist() for bv in self.b]
            })
        except Exception as e:
            print(f"[ContextBandit] Failed to save models: {e}")

    def pull(self, context: np.ndarray) -> int:
        """Select the best arm given current context using LinUCB algorithm."""
        if context.shape[0] != self.state_dim:
            return random.randrange(self.num_arms)
        
        x = context.reshape((self.state_dim, 1)).astype(float)
        scores = np.zeros(self.num_arms, dtype=float)
        
        for i in range(self.num_arms):
            try:
                A_inv = np.linalg.inv(self.A[i])
                theta = A_inv @ self.b[i]
                pred = float((theta.T @ x).squeeze())
                bonus = float(np.sqrt((x.T @ A_inv @ x).squeeze()))
                scores[i] = pred + self.alpha * bonus
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular
                scores[i] = 0.0
                
        return int(np.argmax(scores))

    def update(self, arm_index: int, reward: float, context: np.ndarray):
        """Update bandit model with observed reward for the selected arm."""
        if context.shape[0] != self.state_dim: 
            return
        if not (0 <= arm_index < self.num_arms): 
            return
            
        x = context.reshape((self.state_dim, 1)).astype(float)
        self.A[arm_index] = (self.A[arm_index] + (x @ x.T)).astype(np.float64, copy=False)
        self.b[arm_index] = cast(Any, (self.b[arm_index] + float(reward) * x).astype(np.float64, copy=False))
        
        # Periodic save
        try:
            if sum(a.trace() for a in self.A) % 20 < 1:
                self.save()
        except Exception:
            pass


class NoOpWorldModel:
    """Graceful fallback when torch is unavailable or the WM isn't ready."""
    
    def __init__(self):
        self.available = False
        self.ready = False
        
    async def imagine_with_policy(self, *args, **kwargs):
        """No-op imagination method."""
        return []
        
    def score_transition(self, state, action):
        """No-op transition scoring."""
        return 0.0


# Optional torch-based world model
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    TORCH_AVAILABLE = True
    
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE and torch is not None:
    class TorchStateVAEWorldModel:
        """
        A world model using a Variational Autoencoder (VAE) to learn a compressed
        latent representation of the state space, and an RNN to model state transitions.
        This allows for efficient planning through imagination.
        """
        
        def __init__(self, input_dim, action_dim, latent_dim=32, rnn_hidden_dim=256):
            self.input_dim = input_dim
            self.action_dim = action_dim
            self.latent_dim = latent_dim
            self.rnn_hidden_dim = rnn_hidden_dim
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
            self.available = True
            
            # VAE Components
            self.encoder = nn.Sequential(  # type: ignore
                nn.Linear(input_dim, 128), nn.ReLU(),  # type: ignore
                nn.Linear(128, 64), nn.ReLU()  # type: ignore
            )
            self.fc_mu = nn.Linear(64, latent_dim)  # type: ignore
            self.fc_logvar = nn.Linear(64, latent_dim)  # type: ignore
            
            self.decoder = nn.Sequential(  # type: ignore
                nn.Linear(latent_dim, 64), nn.ReLU(),  # type: ignore
                nn.Linear(64, 128), nn.ReLU(),  # type: ignore
                nn.Linear(128, input_dim)  # type: ignore
            )
            
            # RNN for transition modeling
            self.transition_rnn = nn.GRU(latent_dim + action_dim, rnn_hidden_dim, batch_first=True)  # type: ignore
            self.transition_output = nn.Linear(rnn_hidden_dim, latent_dim)  # type: ignore
            
            self.ready = False

        def encode(self, x):
            """Encode state to latent distribution parameters."""
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

        def decode(self, z):
            """Decode latent representation back to state space."""
            return self.decoder(z)

        def reparameterize(self, mu, logvar):
            """Reparameterization trick for VAE."""
            std = torch.exp(0.5 * logvar)  # type: ignore
            eps = torch.randn_like(std)  # type: ignore
            return mu + eps * std

        async def imagine_with_policy(self, initial_state, policy, horizon=5):
            """Imagine future trajectory using learned world model."""
            if not self.ready:
                return []
                
            trajectory = []
            state = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0)  # type: ignore
            hidden = None
            
            for step in range(horizon):
                # Encode current state
                mu, logvar = self.encode(state)
                z = self.reparameterize(mu, logvar)
                
                # Get action from policy
                action = await policy.get_action(state.numpy())
                action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)  # type: ignore
                
                # Predict next latent state
                rnn_input = torch.cat([z, action_tensor], dim=-1).unsqueeze(1)  # type: ignore
                rnn_output, hidden = self.transition_rnn(rnn_input, hidden)
                next_z = self.transition_output(rnn_output.squeeze(1))
                
                # Decode to state space
                next_state = self.decode(next_z)
                
                trajectory.append({
                    'state': state.numpy(),
                    'action': action,
                    'next_state': next_state.numpy(),
                    'latent': z.detach().numpy()
                })
                
                state = next_state
                
            return trajectory

        def score_transition(self, state, action):
            """Score a state-action transition for planning."""
            if not self.ready:
                return 0.0
                
            try:
                state_tensor = torch.tensor(state, dtype=torch.float32)  # type: ignore
                mu, logvar = self.encode(state_tensor.unsqueeze(0))
                # Simple scoring based on latent variance (uncertainty)
                uncertainty = torch.exp(0.5 * logvar).mean().item()  # type: ignore
                return float(1.0 / (1.0 + uncertainty))
            except Exception:
                return 0.0

else:
    # Fallback when torch is not available
    class TorchStateVAEWorldModel(NoOpWorldModel):
        def __init__(self, *args, **kwargs):
            super().__init__()
