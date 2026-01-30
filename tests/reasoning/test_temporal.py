#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
import pytest
from jlnn.reasoning.temporal import AlwaysNode, EventuallyNode

class MockNode:
    """Simple mock node that returns predefined truth values for testing."""
    def __init__(self, data):
        self.data = data
    def forward(self, values):
        return self.data

def test_always_node_logic():
    """
    Tests the 'Always' (G) operator using Gödel's t-norm (minimum).
    
    Verifies that the node correctly identifies the 'worst-case' truth 
    interval across a sequence.
    """
    # Input shape: (batch=1, time=3, interval=2)
    # Sequence: [0.9, 1.0], [0.5, 0.8], [0.7, 0.9] -> Min should be [0.5, 0.8]
    sequence = jnp.array([[[0.9, 1.0], [0.5, 0.8], [0.7, 0.9]]])
    child = MockNode(sequence)
    node = AlwaysNode(child)
    
    output = node.forward({})
    assert jnp.allclose(output, jnp.array([[0.5, 0.8]]))

def test_eventually_node_logic():
    """
    Tests the 'Eventually' (F) operator using the maximum t-conorm.
    
    Verifies that the node correctly identifies the 'best-case' truth 
    interval across a sequence.
    """
    # Sequence: [0.1, 0.2], [0.8, 0.9], [0.3, 0.4] -> Max should be [0.8, 0.9]
    sequence = jnp.array([[[0.1, 0.2], [0.8, 0.9], [0.3, 0.4]]])
    child = MockNode(sequence)
    node = EventuallyNode(child)
    
    output = node.forward({})
    assert jnp.allclose(output, jnp.array([[0.8, 0.9]]))