#!/usr/bin/env python3
import numpy as np
import asyncio
from memory.manager import MemoryManager

def main():
    print("creating MemoryManager")
    m = MemoryManager(embed_dim=16, seed=1)
    ids = ['a', 'b', 'c']
    rs = np.random.RandomState(1)
    vecs = rs.randn(3, 16).astype('float32')
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    m.build_index(ids, vecs)
    print("index_ready=", m.is_index_ready())

    ratings = [0.9, 0.5, 0.2]
    print("updating hopfield prototypes")
    try:
        m.hopfield.update_prototypes([v for v in vecs], ratings, top_k=2)
        print("update_prototypes ok")
    except Exception as e:
        print("update_prototypes err", e)

    v = vecs[0]
    try:
        cleaned = m.hopfield.clean(v, steps=3, tau=0.1)
        print("hopfield_cleaned_norm=", np.linalg.norm(cleaned))
    except Exception as e:
        print("hopfield_clean_err", e)

    try:
        m.sdm.write(v, v)
        rs_val = m.sdm.read_strength(v)
        rvec = m.sdm.read(v, k=8)
        print("sdm_read_strength=", rs_val, " sdm_read_norm=", np.linalg.norm(rvec))
    except Exception as e:
        print("sdm_err", e)

    for i, id0 in enumerate(ids):
        m.graph.add_node(id0, temperature=1.0, rating=ratings[i])

    # rebuild index to ensure consistency
    m.build_index(ids, vecs)

    res = asyncio.get_event_loop().run_until_complete(
        m.synthesize_remnant(ids, label_hint='test', llm_client=None, is_macro=False)
    )
    print("synthesize_remnant id=", res[0], " mass=", res[2], " vec_norm=", np.linalg.norm(res[1]))


if __name__ == "__main__":
    main()
