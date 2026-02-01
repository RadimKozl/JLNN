#!/usr/bin/env python3
"""
Unit tests for StableHLO (OpenXLA) export.

Verifies that JLNN models can be lowered into hardware-agnostic 
MLIR modules for high-performance deployment.
"""

# Imports
import pytest
import jax.numpy as jnp
from flax import nnx
from jlnn.nn.gates import WeightedAnd
from jlnn.export.stablehlo import export_to_stablehlo

def test_stablehlo_roundtrip_accuracy():
    """
    Ensures numerical consistency after StableHLO compilation.
    
    Compares the output of the original JAX model with the 
    compiled HLO artifact to ensure no precision was lost.
    """
    rngs = nnx.Rngs(0)
    model = WeightedAnd(num_inputs=2, rngs=rngs)
    
    sample_input = jnp.array([[0.2, 0.8], [0.5, 0.5]]) # (batch, 2)
    
    # Export
    exported = export_to_stablehlo(model, sample_input)
    
    # Capture original output
    original_out = model(sample_input)
    
    # Run exported (StableHLO)
    # We split to get the current state for the call
    _, state = nnx.split(model)
    exported_out = exported.call(state, sample_input)
    
    assert jnp.allclose(original_out, exported_out, atol=1e-5), \
        "StableHLO output differs from original JAX output"

def test_stablehlo_mlir_generation():
    """
    Inspects the generated MLIR code.
    
    Checks if the exported module contains expected HLO 
    dialects and logical operation patterns.
    """
    rngs = nnx.Rngs(42)
    model = WeightedAnd(num_inputs=2, rngs=rngs)
    sample = jnp.zeros((1, 2))
    
    exported = export_to_stablehlo(model, sample)
    mlir_code = exported.mlir_module()
    
    assert "stablehlo.add" in mlir_code or "stablehlo.compare" in mlir_code
    assert "main" in mlir_code