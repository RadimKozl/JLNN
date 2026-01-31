#!/usr/bin/env python3
"""
Unit tests for xarray integration.

Ensures that JAX tensors are correctly labeled and transformed into 
xarray structures, and verifies compatibility with modern xarray APIs.
"""

# Imports
import jax.numpy as jnp
import xarray as xr
from jlnn.utils.xarray_utils import model_to_xarray, extract_weights_to_xarray

def test_model_to_xarray_structure():
    """
    Verifies that gate outputs are correctly labeled and mapped to samples.
    
    This test uses .sizes instead of .dims to comply with modern xarray API 
    and avoid FutureWarnings.
    """
    outputs = {
        "gate_and": jnp.array([[0.1, 0.9], [0.2, 0.8]]),
        "gate_or": jnp.array([[0.5, 0.5], [0.0, 1.0]])
    }
    sample_names = ["Patient_1", "Patient_2"]
    
    ds = model_to_xarray(outputs, sample_names)
    
    assert isinstance(ds, xr.Dataset)
    assert "gate_and" in ds
    # Fix for FutureWarning: use .sizes instead of .dims
    assert ds.sizes["bound"] == 2
    assert ds.sizes["sample"] == 2
    assert list(ds.coords["bound"].values) == ["Lower", "Upper"]
    assert ds.coords["sample"].values[0] == "Patient_1"

def test_extract_weights_to_xarray():
    """
    Ensures that logical weights are correctly mapped to symbolic input names.
    """
    weights = jnp.array([1.2, 4.5])
    labels = ["is_mammal", "has_legs"]
    
    da = extract_weights_to_xarray(weights, labels, "Gate1")
    
    assert isinstance(da, xr.DataArray)
    assert da.sizes["input"] == 2
    assert da.coords["input"].values[1] == "has_legs"
    assert jnp.allclose(da.values, weights)