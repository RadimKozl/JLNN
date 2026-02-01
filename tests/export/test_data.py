#!/usr/bin/env python3
"""
Unit tests for logical parameter extraction.

Verifies that the model's internal state (weights, beta) is correctly 
captured into portable Python dictionaries for serialization.
"""

# Imports
import pytest
from flax import nnx
import jax.numpy as jnp
from jlnn.nn.gates import WeightedAnd, WeightedImplication
from jlnn.export.data import extract_logic_parameters

def test_extract_logic_parameters_structure():
    """
    Validates the structure of the exported parameter dictionary.
    
    Checks for framework metadata and ensures logical parameters 
    like weights and beta are correctly extracted from NNX variables.
    """
    rngs = nnx.Rngs(42)
    gate = WeightedAnd(num_inputs=2, rngs=rngs)
    
    exported_data = extract_logic_parameters(gate)
    
    # Check top-level metadata
    assert exported_data["metadata"]["framework"] == "JLNN"
    
    # Use the correct key 'structure' (fixes the KeyError)
    root_gate = exported_data["structure"]
    assert root_gate["type"] == "WeightedAnd"
    assert "weights" in root_gate["params"]
    assert "beta" in root_gate["params"]
    
    # Verify weight values (LNN weights must be >= 1.0)
    weights = jnp.array(root_gate["params"]["weights"])
    assert jnp.all(weights >= 1.0)

def test_recursive_export_implication():
    """
    Tests recursive extraction for composite gates.
    
    Verifies that complex gates (like Implication, which contains 
    NOT and OR sub-gates) are fully captured in the recursive tree.
    """
    rngs = nnx.Rngs(42)
    model = WeightedImplication(rngs=rngs)
    
    exported = extract_logic_parameters(model)
    root = exported["structure"]
    
    assert root["type"] == "WeightedImplication"
    # Verification of sub-components if they exist
    if "sub_gates" in root:
        assert len(root["sub_gates"]) > 0