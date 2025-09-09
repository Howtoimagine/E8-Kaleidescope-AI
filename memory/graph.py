"""
Graph Database Management for E8Mind Memory System
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

try:
    import networkx as nx
    from networkx.algorithms import community as nx_comm
    from networkx.readwrite import json_graph
except Exception:
    nx = None
    nx_comm = None
    
    class _JsonGraph:
        def node_link_data(self, g): 
            return {"nodes": [], "links": []}
        def node_link_graph(self, d): 
            return None
    json_graph = _JsonGraph()

class GraphDB:
    """A graph database wrapper around NetworkX for managing conceptual relationships."""
    
    def __init__(self):
        if nx is None: 
            raise ImportError("networkx library is required for GraphDB.")
        self.graph = nx.Graph()
    
    def add_node(self, node_id: str, **attrs):
        """Adds a node to the graph with the given attributes."""
        self.graph.add_node(node_id, **attrs)
    
    def add_edge(self, source_id: str, target_id: str, **attrs):
        """Adds an edge between two nodes with the given attributes."""
        self.graph.add_edge(source_id, target_id, **attrs)
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a node's data."""
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]
        return None
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Gets the neighbors of a node."""
        if self.graph.has_node(node_id):
            return list(self.graph.neighbors(node_id))
        return []

    def compute_and_store_communities(self, partition_key: str = "community_id"):
        """Computes Louvain communities and stores the partition ID on each node."""
        if nx_comm is None or self.graph.number_of_nodes() < 10:
            return
        try:
            from typing import cast, Iterable
            seed = int(os.getenv("E8_SEED", "42"))
            communities_iter = nx_comm.louvain_communities(self.graph, seed=seed)
            communities = list(cast(Iterable, communities_iter))
            for i, community_nodes in enumerate(communities):
                for node_id in community_nodes:
                    if self.graph.has_node(node_id):
                        self.graph.nodes[node_id][partition_key] = i
            print(f"[GraphDB] Computed {len(communities)} communities.")
        except Exception as e:
            print(f"[GraphDB] Community detection failed: {e}")

    def increment_edge_weight(self, u, v, delta=0.1, min_w=0.0, max_w=10.0, **attrs):
        """Create edge if absent; add delta to 'weight' clamped to [min_w, max_w]."""
        try:
            if not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v, weight=max(min_w, delta), **attrs)
            else:
                w = float(self.graph.get_edge_data(u, v, default={'weight': 0.0}).get('weight', 0.0)) + float(delta)
                w = min(max_w, max(min_w, w))
                self.graph[u][v]['weight'] = w
                for k, val in attrs.items():
                    self.graph[u][v][k] = val
        except Exception as e:
            try: 
                print(f"[GraphDB] increment_edge_weight failed: {e}")
            except Exception: 
                pass
