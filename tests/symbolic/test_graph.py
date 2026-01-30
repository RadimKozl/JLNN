#!/usr/bin/env python3

# Imports
import networkx as nx
from jlnn.symbolic.compiler import LNNFormula
from jlnn.symbolic.graph import build_networkx_graph, from_networkx_to_jlnn

def test_graph_conversion(rngs):
    """
    Tests the bidirectional conversion between JLNN models and NetworkX graphs.
    """
    model = LNNFormula("A | B", rngs)
    
    # Export to NetworkX
    graph = build_networkx_graph(model.root)
    
    assert isinstance(graph, nx.DiGraph)
    # We expect 3 nodes: Predicate A, Predicate B, and the OR Gate
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2

def test_graph_reconstruction(rngs):
    """
    Verifies that a JLNN model can be reconstructed from a NetworkX graph structure.
    """
    model = LNNFormula("A & B", rngs)
    graph = build_networkx_graph(model.root)
    
    # Reconstruct
    root_id = id(model.root)
    new_root = from_networkx_to_jlnn(graph, root_id, rngs)
    
    assert new_root is not None
    # Check if the reconstructed root is an N-Ary gate (the AND gate)
    from jlnn.symbolic.compiler import NAryGateNode
    assert isinstance(new_root, NAryGateNode)