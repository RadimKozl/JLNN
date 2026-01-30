#!/usr/bin/env python3
"""
Unit tests for the JLNNEngine class.
Ensures JIT compilation and inference work correctly within the NNX framework.
"""

import jax.numpy as jnp
from jlnn.reasoning.engine import JLNNEngine
from jlnn.symbolic.compiler import LNNFormula

def test_engine_jit_inference(rngs):
    """
    Verifies that the engine performs JIT-compiled inference and handles shapes correctly.
    
    The output is expected to be (batch, sequence, 2).
    """
    formula = "A & B"
    model = LNNFormula(formula, rngs)
    engine = JLNNEngine(model)
    
    # Input: 1 batch, 1 time-step
    inputs = {
        "A": jnp.array([[1.0]]),
        "B": jnp.array([[1.0]])
    }
    
    output = engine.infer(inputs)
    
    # The model preserves the sequence dimension: (batch=1, time=1, interval=2)
    assert output.shape == (1, 1, 2)
    assert output[0, 0, 0] > 0.5  # Check lower bound of the AND result