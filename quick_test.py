#!/usr/bin/env python3
"""
Quick test script to verify the monolith loads and basic paths work.
This script avoids non-ASCII output to play nice with Windows consoles.
"""

import asyncio
import importlib.util


class MockConsole:
    def log(self, msg):
        try:
            print(f"[LOG] {msg}")
        except Exception:
            print("[LOG]", str(msg).encode('ascii', errors='replace').decode('ascii'))
    def rule(self, msg):
        try:
            print(f"[RULE] {msg}")
        except Exception:
            print("[RULE]", str(msg).encode('ascii', errors='replace').decode('ascii'))


class MockMind:
    def __init__(self):
        self.step_num = 0
        self.console = MockConsole()

    async def rate_concept(self, text):
        return 0.5


class MockMemory:
    def __init__(self):
        import networkx as nx
        self.graph_db = self
        self.graph = nx.Graph()
        self.main_vectors = {}
        self.label_to_node_id = {}
        self.pending_additions = []
        self.pending_embeddings = []
        self.lock = asyncio.Lock()

    def get_node(self, node_id):
        return self.graph.nodes.get(node_id)

    def add_node(self, node_id, **kwargs):
        self.graph.add_node(node_id, **kwargs)

    def add_edge(self, src, dst, **kwargs):
        self.graph.add_edge(src, dst, **kwargs)

    async def add_entry(self, entry_data):
        import hashlib
        import time
        node_id = entry_data.get('idx')
        if not node_id:
            content_str = f"{entry_data.get('label', '')}{entry_data.get('metaphor', '')}{time.time()}"
            node_id = hashlib.sha1(content_str.encode()).hexdigest()[:16]
            entry_data['idx'] = node_id
        if self.graph.has_node(node_id):
            return node_id
        self.graph.add_node(node_id, **entry_data)
        return node_id


async def test_concept_addition():
    print("\nTesting concept addition...")
    mock_mind = MockMind()
    mock_memory = MockMemory()
    mock_mind.memory = mock_memory

    def sanitize_line(text, max_len):
        return text[:max_len] if len(text) > max_len else text

    def sanitize_block(text, max_lines, max_chars):
        return text[:max_chars] if len(text) > max_chars else text

    test_text = "Quantum entanglement demonstrates non-local correlations between particles."
    entry = {
        "type": "external_concept",
        "label": sanitize_line(test_text, 40),
        "metaphor": sanitize_block(test_text, 5, 500),
        "rating": 0.8,
        "step": 0,
        "source": "test_source",
    }

    concept_id = await mock_memory.add_entry(entry)
    print(f"Added concept with ID: {concept_id}")

    node_count = mock_memory.graph.number_of_nodes()
    print(f"Nodes after addition: {node_count}")

    if node_count > 0:
        for node_id in mock_memory.graph.nodes():
            node_data = mock_memory.graph.nodes[node_id]
            print(f"  Node {node_id}: {node_data.get('label', 'No label')}")
        return True
    else:
        print("No nodes were created")
        return False


async def test_data_sources():
    print("\nTesting data sources...")
    path = 'e8_mind_server_M18.7.py'
    spec = importlib.util.spec_from_file_location('e8monolith', path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore
    except Exception as e:
        print(f"Import error: {e}")
        return False
    ds = getattr(mod, 'DATA_SOURCES', {})
    if isinstance(ds, dict) and ds:
        print("Found DATA_SOURCES definition")
        print(f"  Keys: {list(ds.keys())}")
        return 'ai_ml_arxiv' in ds and 'physics_arxiv' in ds
    print("Could not find DATA_SOURCES")
    return False


async def main():
    print("E8Mind Quick Test")
    print("=" * 50)

    try:
        test1 = await test_data_sources()
        test2 = await test_concept_addition()
        if test1 and test2:
            print("\nAll tests passed!")
        else:
            print("\nSome tests failed")
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

