import json
import numpy as np
from fastapi.testclient import TestClient

from web.server import create_app

def norm8():
    v = np.random.randn(8).astype("float32")
    v /= np.linalg.norm(v) + 1e-12
    return v.tolist()

def main():
    app = create_app()
    client = TestClient(app)

    # Health
    r = client.get("/api/health")
    print("health", r.status_code, r.json())

    # Blueprint
    r = client.get("/api/blueprint")
    bp = r.json()
    print("blueprint", r.status_code, len(bp.get("points", [])))

    # Sample
    anchors = [{"vector": norm8(), "weight": 0.5}, {"vector": norm8(), "weight": 0.5}]
    r = client.post("/api/sample", json={"anchors": anchors, "top_k": 3, "weyl_draws": 0})
    smp = r.json()
    print("sample", r.status_code, smp.get("items", [])[:3])

    # Potential
    r = client.post("/api/potential", json={"anchors": anchors, "kernel": "cosine", "rbf_sigma": 0.8})
    pot = r.json()
    vals = pot.get("values", [])
    print("potential", r.status_code, len(vals), (min(vals) if vals else None), (max(vals) if vals else None))

    # Consolidate (single 1536-d vector)
    emb_dim = app.state.services.memory.embed_dim
    vec = np.random.randn(emb_dim).astype("float32").tolist()
    r = client.post("/api/consolidate", json={"vectors": [vec]})
    cons = r.json()
    out = cons.get("vectors", [[]])
    print("consolidate", r.status_code, len(out), (len(out[0]) if out and out[0] else 0))

if __name__ == "__main__":
    main()
