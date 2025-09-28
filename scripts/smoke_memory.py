import sys
import os
import numpy as np

# Ensure workspace root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from e8_mind.memory import MemoryManager, VariationalAutoencoder


def main():
    mm = MemoryManager()
    sh = mm.ensure_shell(8, mind_instance=mm)
    for i in range(10):
        sh.add_vector(f"n{i}", np.random.randn(8).astype("float32"))

    prox = mm.attach_proximity()
    prox.update_shell_index(8, sh)
    q = np.random.randn(8).astype("float32")
    res = prox.knn(8, sh, q, k=3)
    print("KNN results:", res)

    print("GraphDB present:", bool(mm.graph_db))
    if mm.graph_db:
        mm.graph_db.add_node("a")
        mm.graph_db.add_node("b")
        mm.graph_db.add_edge("a", "b", weight=1.5)
        pa = prox.shortest_path(mm, "a", "b")
        print("PathAsset:", pa)

    vae = VariationalAutoencoder()
    z = vae.project_to_dim(np.random.randn(16).astype("float32"), 8)
    print("Projected len:", len(np.asarray(z).reshape(-1)))


if __name__ == "__main__":
    main()
