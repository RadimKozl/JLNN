#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
import pytest
from jlnn.reasoning.temporal import AlwaysNode, EventuallyNode

class MockNode:
    """
    A minimal mock object simulating a logical node for isolated unit testing.
    
    Used to inject specific truth intervals into temporal operators without 
    requiring a full model compilation.
    """
    def __init__(self, data: jnp.ndarray):
        """
        Initializes the mock node with static data.

        Args:
            data: A JAX array of truth intervals, typically with shape 
                (batch, time, 2) or (batch, 2).
        """
        self.data = data
        
    def forward(self, values: dict) -> jnp.ndarray:
        """
        Returns the pre-defined truth intervals.

        Args:
            values: Dictionary of inputs (ignored by the mock).

        Returns:
            The JAX array provided during initialization.
        """
        return self.data

def test_always_node_logic():
    """
    Tests the 'Always' (Globally) operator semantics using the Gödel t-norm.
    
    Formula: G(phi)
    The operator must perform a minimum-based aggregation across the time axis.
    Input sequence: [0.9, 1.0], [0.5, 0.8], [0.7, 0.9]
    Expected result: [0.5, 0.8] (the intersection/minimum truth).
    """
    # Shape: (batch=1, time=3, interval=2)
    sequence = jnp.array([[[0.9, 1.0], [0.5, 0.8], [0.7, 0.9]]])
    child = MockNode(sequence)
    node = AlwaysNode(child)
    
    output = node.forward({})
    assert jnp.allclose(output, jnp.array([[0.5, 0.8]]))

def test_eventually_node_logic():
    """
    Tests the 'Eventually' (Finally) operator semantics using the t-conorm.
    
    Formula: F(phi)
    The operator must perform a maximum-based aggregation across the time axis.
    Input sequence: [0.1, 0.2], [0.4, 0.6], [0.2, 0.3]
    Expected result: [0.4, 0.6] (the highest truth achieved in the sequence).
    """
    # Shape: (batch=1, time=3, interval=2)
    sequence = jnp.array([[[0.1, 0.2], [0.8, 0.9], [0.3, 0.4]]])
    child = MockNode(sequence)
    node = EventuallyNode(child)
    
    output = node.forward({})
    assert jnp.allclose(output, jnp.array([[0.8, 0.9]]))