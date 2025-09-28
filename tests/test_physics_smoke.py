import os
import numpy as np

from e8_mind.physics.e8 import E8Physics
from e8_mind.physics.horizon import HorizonLayer, build_e8_horizon, build_cross_horizon_kernel
from e8_mind.physics.quantum import QuantumConfig, QuantumEngine


class Console:
    def log(self, *a, **k):
        pass


def test_e8_physics_init():
    console = Console()
    e8 = E8Physics(console)
    assert e8.roots.shape[0] > 0
    assert e8.L_norm is not None
    # heat mask
    m = e8.heat_mask_cached(0, 1.25)
    assert m.shape[0] == e8.roots.shape[0]


def test_horizon_build_and_kernel():
    console = Console()
    e8 = E8Physics(console)
    # Make a basic "blueprint": project roots to 3D via fixed matrix
    P = e8.roots[:, :3]
    # Construct fake edges among first 10 points
    edges = [(i, (i+1) % 10) for i in range(10)]
    H_e8 = build_e8_horizon(e8, P, edges, console)
    # Shell horizon with slight jitter
    shell = HorizonLayer("H_shell_3")
    shell.pos = P[:20] + 0.01 * np.random.default_rng(0).standard_normal(P[:20].shape)
    K = build_cross_horizon_kernel(H_e8, shell)
    # K can be sparse matrix or dict fallback
    if hasattr(K, 'shape'):
        assert K.shape[0] == shell.pos.shape[0]
    else:
        assert isinstance(K, dict)


def test_quantum_engine_step_and_measure():
    console = Console()
    e8 = E8Physics(console)
    cfg = QuantumConfig(dt=0.1)
    q = QuantumEngine(e8, cfg, console)
    tv = q.step_adaptive()
    assert isinstance(tv, float)
    choices = q.measure_hybrid(prev_idx=0)
    assert len(choices) == cfg.batch
