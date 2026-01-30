#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.reasoning.engine import JLNNEngine
from jlnn.symbolic.compiler import LNNFormula

def test_engine_jit_inference(rngs):
    """
    Verifies that the JLNNEngine correctly performs JIT-compiled inference.
    
    This test ensures:
    1. The engine successfully wraps an LNNFormula (nnx.Module).
    2. JAX compilation (JIT) is triggered and executes without errors.
    3. The output preserves the temporal/batch dimensions: (batch, sequence, 2).
    4. Basic logical truth is maintained through the neural execution path.
    """
    formula = "A & B"
    model = LNNFormula(formula, rngs)
    engine = JLNNEngine(model)
    
    # Input: 1 batch, 1 time-step with full truth [1.0, 1.0]
    inputs = {
        "A": jnp.array([[1.0]]),
        "B": jnp.array([[1.0]])
    }
    
    output = engine.infer(inputs)
    
    # Validation of output structure and logic
    assert output.shape == (1, 1, 2)
    assert output[0, 0, 0] > 0.5  # Check lower bound of the AND result