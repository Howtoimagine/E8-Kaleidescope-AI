import pytest

from memory.manager import MemoryManager

def test_graph_db_alias_presence():
    mm = MemoryManager(embed_dim=16, seed=1)
    assert hasattr(mm, 'graph'), 'MemoryManager missing primary graph attribute'
    assert hasattr(mm, 'graph_db'), 'MemoryManager missing backward-compat graph_db alias'
    # Both should reference the same object
    assert mm.graph is mm.graph_db, 'graph_db alias does not reference the same GraphDB instance'

    # Basic node add via either reference should reflect in both
    mm.graph.add_node('n1', label='node1')
    assert 'n1' in mm.graph_db.graph, 'Node added via graph not visible through graph_db'
    mm.graph_db.add_node('n2', label='node2')
    assert 'n2' in mm.graph.graph, 'Node added via graph_db not visible through graph'
