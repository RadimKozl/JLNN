#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.nn import gates
from jlnn.core import intervals

def test_weighted_or_gate_forward(rngs):
    """
    Verifies the forward pass and output shape of the WeightedOr gate.

    This test ensures that:
    1. The gate correctly processes a batch of interval inputs.
    2. The resulting truth values remain within the valid [0, 1] logical range.
    3. The module's internal parameter access ([...]) works seamlessly 
       within the Flax NNX framework.

    Args:
        rngs (nnx.Rngs): Flax NNX random number generator collection.
    """
    batch_size = 5
    n_in = 2
    gate = gates.WeightedOr(n_in, rngs)
    
    # Input data representing medium uncertainty: [0.5, 0.5]
    x = jnp.ones((batch_size, n_in, 2)) * 0.5
    res = gate(x)
    
    # Assert output dimensions (batch_size, 2) where 2 is the [L, U] pair
    assert res.shape == (batch_size, 2)
    # Logical consistency: values must be in [0, 1]
    assert jnp.all(res >= 0.0) and jnp.all(res <= 1.0)

def test_weighted_implication_gate(rngs):
    """
    Tests the semantic correctness of the WeightedImplication gate.

    Focuses on the classical Łukasiewicz boundary condition: 
    If the antecedent (A) is fully True (1.0) and the consequent (B) 
    is fully False (0.0), the implication (A -> B) must be fully False.

    Args:
        rngs (nnx.Rngs): Flax NNX random number generator collection.
    """
    gate = gates.WeightedImplication(rngs, method='lukasiewicz')
    
    # Define A as [1.0, 1.0] (True) and B as [0.0, 0.0] (False)
    a = intervals.create_interval(jnp.array([1.0]), jnp.array([1.0]))
    b = intervals.create_interval(jnp.array([0.0]), jnp.array([0.0]))
    
    # Forward pass: 1 -> 0 must result in 0
    res = gate(a, b)
    
    # Check the upper bound of the result
    assert jnp.isclose(intervals.get_upper(res), 0.0)