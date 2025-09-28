import os, sys
import numpy as np

# Ensure repository root is on sys.path for local package imports
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from e8_mind.physics.e8 import E8Physics
from e8_mind.physics.quantum import QuantumConfig, QuantumEngine

try:
    from e8_mind.physics.horizon import HorizonLayer, build_e8_horizon, build_cross_horizon_kernel
    HAVE_HORIZON = True
except Exception:
    HAVE_HORIZON = False


class Console:
    def log(self, *a, **k):
        pass


def main():
    console = Console()
    e8 = E8Physics(console)
    assert e8.roots.shape[0] == 240
    print('[SMOKE] E8 init OK')

    if HAVE_HORIZON:
        try:
            P = e8.roots[:, :3]
            edges = [(i, (i + 1) % 10) for i in range(10)]
            H_e8 = build_e8_horizon(e8, P, edges, console)
            shell = HorizonLayer('H_shell_3')
            shell.pos = P[:20] + 0.01 * np.random.default_rng(0).standard_normal(P[:20].shape)
            K = build_cross_horizon_kernel(H_e8, shell)
            print('[SMOKE] Horizon OK; K type:', type(K).__name__)
        except Exception as e:
            print('[SMOKE] Horizon skipped:', e)
    else:
        print('[SMOKE] Horizon module unavailable; skipped')

    cfg = QuantumConfig(dt=0.1)
    q = QuantumEngine(e8, cfg, console)
    tv = q.step_adaptive()
    choices = q.measure_hybrid(prev_idx=0)
    print('[SMOKE] Quantum OK; tv=%.4f, choices=%d' % (float(tv), len(choices)))


if __name__ == '__main__':
    main()
