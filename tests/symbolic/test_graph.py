#!/usr/bin/env python3

# Imports
import networkx as nx
from jlnn.symbolic.compiler import LNNFormula
from jlnn.symbolic.graph import build_networkx_graph, from_networkx_to_jlnn

def test_graph_conversion(rngs):
    """
    Tests the conversion of a JLNN model structure into a NetworkX directed graph.
    
    Verifies that the logical hierarchy is correctly mapped to graph nodes and edges,
    allowing for topological analysis or visualization.
    """
    model = LNNFormula("A | B", rngs)
    
    # Export the model tree starting from the root node to NetworkX
    graph = build_networkx_graph(model.root)
    
    assert isinstance(graph, nx.DiGraph)
    # Expecting 3 nodes: Predicate A, Predicate B, and the OR gate node
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2

def test_graph_reconstruction(rngs):
    """
    Verifies that a JLNN model can be accurately reconstructed from a NetworkX graph.
    
    This ensures "round-trip" integrity: a formula can be converted to a graph 
    and back into a functional JLNN structure without losing its logical type or connectivity.
    """
    model = LNNFormula("A & B", rngs)
    graph = build_networkx_graph(model.root)
    
    # Reconstruct the JLNN node structure from the graph data
    root_id = id(model.root)
    new_root = from_networkx_to_jlnn(graph, root_id, rngs)
    
    assert new_root is not None
    # Verify that the reconstructed root node maintains its logical identity (AND gate)
    from jlnn.symbolic.compiler import NAryGateNode
    assert isinstance(new_root, NAryGateNode)