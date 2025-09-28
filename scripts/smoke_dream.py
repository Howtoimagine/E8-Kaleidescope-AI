import asyncio
import os
import sys

# Make repo root importable
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from e8_mind.cognitive.dream import DreamEngine, DreamReplayService


class FakeConsole:
    def print(self, *args, **kwargs):
        pass
    def log(self, *args, **kwargs):
        pass

class FakeGraphDB:
    def __init__(self):
        import networkx as nx
        self.graph = nx.Graph()
        # seed two nodes and an edge with minimal attributes
        self.graph.add_node('A', type='concept', label='Alpha', step=0)
        self.graph.add_node('B', type='concept', label='Beta', step=1)
        self.graph.add_edge('A','B', weight=1.0)
    def get_neighbors(self, nid):
        return list(self.graph.neighbors(nid))
    def get_node(self, nid):
        return self.graph.nodes.get(nid)
    def increment_edge_weight(self, u,v, delta=0.1, kind="consolidated"):
        if self.graph.has_edge(u,v):
            w = self.graph[u][v].get('weight', 0.0)
            self.graph[u][v]['weight'] = w + delta

class FakeMemory:
    def __init__(self):
        self.graph_db = FakeGraphDB()
        self.main_vectors = {'A':[0.1,0.2], 'B':[0.2,0.4]}
        async def add_entry(doc, parent_ids=None):
            return 'X'
        self.add_entry = add_entry
        self.vsa = type('V', (), {'encode_parentage': lambda *args, **kwargs: [0.0,0.0]})()

class FakePrompts:
    def render(self, name, **kwargs):
        return f"Run thought experiment on {kwargs.get('concept','unknown')}"

class FakeLLM:
    async def enqueue_and_wait(self, prompt, max_tokens=128, temperature=0.0):
        return "A short hypothetical narrative about Alpha and Beta."

class FakeGoals:
    def get_top_goals(self, k=1):
        return [(1, 'achieve better understanding')]

class FakeMind:
    def __init__(self):
        self.console = FakeConsole()
        self.prompts = FakePrompts()
        self.llm_pool = FakeLLM()
        self.goal_field = FakeGoals()
        self.step_num = 0
        self.metrics = type('M', (), {'increment': lambda *a, **k: None, 'timing': lambda *a, **k: None, 'observe': lambda *a, **k: None})()
        self.subconscious_event_log = []
        self.memory = FakeMemory()
        self.insight_agent = type('IA', (), {'episodic_memory': type('EM', (), {'sample_prioritized': lambda *a, **k: [{'node_id':'B', 'parent_ids':['A'], 'rating':0.3}]})()})()
        self.action_dim = 2
        self.world_model = type('WM', (), {'available': True, 'train_batch': lambda self, batch: {'loss_recon':0.1, 'loss_kl':0.01}})()

async def main():
    mind = FakeMind()
    de = DreamEngine(mind.memory, mind)
    await de.run_dream_sequence()
    dr = DreamReplayService(mind, batch=1, steps=1)
    await dr.run()
    print('dream smoke ok; events:', len(mind.subconscious_event_log))

if __name__ == '__main__':
    asyncio.run(main())
