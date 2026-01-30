#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.symbolic.compiler import LNNFormula
from jlnn.nn import gates, predicates

def test_compiler_end_to_end(rngs):
    """
    Tests the full compilation pipeline from a formula string to a callable JAX model.
    
    This test verifies:
    1. Successful model initialization and internal structure creation.
    2. The production of a valid truth interval [Lower bound, Upper bound].
    3. Compatibility with the Flax NNX module system.
    """
    formula_str = "A & B -> C"
    model = LNNFormula(formula_str, rngs)
    
    # Prepare mock input data (batch_size=1)
    # Each input feature should be a numeric array representing initial truth values
    inputs = {
        "A": jnp.array([[0.9]]),
        "B": jnp.array([[0.8]]),
        "C": jnp.array([[0.1]])
    }
    
    # Forward pass through the compiled LNN
    output = model(inputs)
    
    # Check shape: (batch_size, 2) where 2 represents the [L, U] interval
    assert output.shape == (1, 2)
    assert jnp.all(output >= 0.0) and jnp.all(output <= 1.0)

def test_compiler_predicate_sharing(rngs):
    """
    Ensures that redundant variables in a formula share the same underlying model weights.
    
    The compiler should detect duplicate variable names (e.g., 'A') and map them 
    to a single LearnedPredicate instance to ensure consistency across the network.
    """
    model = LNNFormula("A & A", rngs)
    # The compiler should store unique predicates in its internal registry
    assert "A" in model.predicates
    assert len(model.predicates) == 1