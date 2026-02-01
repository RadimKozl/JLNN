#!/usr/bin/env python3
"""
Unit tests for ONNX export functionality.
"""

# Imports
import pytest
import jax.numpy as jnp
from flax import nnx
from jlnn.nn.gates import WeightedOr
from jlnn.export.onnx import export_to_onnx

def test_onnx_export_file_creation(tmp_path):
    """
    Verifies that the ONNX exporter successfully writes to disk.
    
    Ensures that the output file exists and contains a non-trivial 
    amount of data after exporting a logical gate.
    """
    onnx_path = tmp_path / "model.onnx"
    rngs = nnx.Rngs(123)
    model = WeightedOr(num_inputs=2, rngs=rngs)
    sample = jnp.array([[0.1, 0.9]])
    
    export_to_onnx(model, sample, str(onnx_path))
    
    assert onnx_path.exists()
    # Check if file size is > 0
    assert onnx_path.stat().st_size > 0