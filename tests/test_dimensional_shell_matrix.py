import numpy as np
import importlib.util, importlib.machinery, os

# Dynamically load the monolith whose filename includes a dot
_monolith_path = os.path.join(os.path.dirname(__file__), '..', 'e8_mind_server_M18.7.py')
_monolith_path = os.path.abspath(_monolith_path)
loader = importlib.machinery.SourceFileLoader('e8_mind_monolith', _monolith_path)
spec = importlib.util.spec_from_loader(loader.name, loader)
if spec is None:
    raise RuntimeError('Failed to create spec for monolith module')
mod = importlib.util.module_from_spec(spec)
loader.exec_module(mod)  # type: ignore
DimensionalShell = getattr(mod, 'DimensionalShell')

class DummyMind:
    def __init__(self):
        self.console = type('C',(),{'log':lambda *a,**k: None})()
    def _snap_to_lattice(self, v, dim):
        v = np.asarray(v, dtype=np.float32)
        if v.shape[0] != dim:
            out = np.zeros(dim, dtype=np.float32)
            n = min(dim, v.shape[0])
            out[:n] = v[:n]
            return out
        return v

def test_empty_shell_matrix():
    shell = DimensionalShell(4, DummyMind())
    M, ids = shell.get_all_vectors_as_matrix()
    assert M is None and ids is None

def test_shell_add_and_matrix():
    shell = DimensionalShell(4, DummyMind())
    shell.add_vector('n1', np.array([1,2,3,4], dtype=np.float32))
    shell.add_vector('n2', np.array([0.1,0.2,0.3,0.4], dtype=np.float32))
    M, ids = shell.get_all_vectors_as_matrix()
    assert M is not None and ids is not None and len(ids)==2
    assert M.shape[0] == 2 and M.shape[1] == 4
