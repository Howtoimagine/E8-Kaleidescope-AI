import numpy as np
from physics.e8_lattice import E8Physics
from physics.engines import QuantumEngine, ClassicalEngine
from neural.autoencoder import SubspaceProjector

class Console:
    def log(self, *args, **kwargs):
        print(*args)

def main():
    console = Console()
    phys = E8Physics(console)

    # Quantum engine
    qe = QuantumEngine(phys, config={"kernel": "cosine", "rbf_sigma": 0.8}, console=console)
    anchors = [(np.random.randn(8).astype("float32"), 0.5),
               (np.random.randn(8).astype("float32"), 0.5)]
    anchors = [(v / (np.linalg.norm(v) + 1e-12), w) for v, w in anchors]
    qe.set_anchors(anchors)
    V = qe.potential(curiosity_visits=np.zeros(240, dtype="float32"), curiosity_alpha=0.1, weyl_draws=0)
    print("potential_shape", V.shape, "min", float(V.min()), "max", float(V.max()))
    print("sample", qe.sample(5))

    # Classical engine
    ce = ClassicalEngine(phys, console=console)
    bp = ce.blueprint()
    print("blueprint_count", len(bp), "first", bp[0] if bp else None)

    # Projector
    proj = SubspaceProjector(seed=123)
    x = np.random.randn(1536).astype("float32")
    y = proj.project_to_dim(x, 8)
    print("projection", y.shape, "norm", float(np.linalg.norm(y)))

if __name__ == "__main__":
    main()
