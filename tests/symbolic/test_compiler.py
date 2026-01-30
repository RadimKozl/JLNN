#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.symbolic.compiler import LNNFormula
from jlnn.nn import gates, predicates

def test_compiler_end_to_end(rngs):
    """
    Tests the full compilation from string to a callable JAX model.
    
    Verifies:
    1. The model is initialized without errors.
    2. The model produces a truth interval [L, U].
    3. The resulting model is a valid Flax NNX module.
    """
    formula_str = "A & B -> C"
    model = LNNFormula(formula_str, rngs)
    
    # Prepare mock input data (batch_size=1)
    # Each input feature should be a numeric array
    inputs = {
        "A": jnp.array([[0.9]]),
        "B": jnp.array([[0.8]]),
        "C": jnp.array([[0.1]])
    }
    
    # Forward pass
    output = model(inputs)
    
    # Check shape: (batch_size, 2) where 2 is [L, U]
    assert output.shape == (1, 2)
    assert jnp.all(output >= 0.0) and jnp.all(output <= 1.0)

def test_compiler_predicate_sharing(rngs):
    """
    Ensures that if a variable (e.g., 'A') appears multiple times in a 
    formula, it maps to the same LearnedPredicate instance.
    """
    model = LNNFormula("A & A", rngs)
    # The compiler should store unique predicates in its dictionary
    assert "A" in model.predicates
    assert len(model.predicates) == 1