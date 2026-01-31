#!/usr/bin/env python3
"""
Unit tests for helper utilities in the JLNN framework.

These tests verify basic data transformation functions that ensure 
compatibility between standard machine learning datasets and the 
interval-based logic of LNNs.
"""

# Imports
import jax.numpy as jnp
from jlnn.utils.helpers import scalar_to_interval, is_precise

def test_scalar_to_interval():
    """
    Verifies the conversion of [0, 1] scalars into [L, U] truth intervals.
    
    In neuro-symbolic grounding, exact values from a dataset are typically 
    treated as precise statements where the lower bound equals the upper bound.
    This test ensures the output tensor has the correct shape and values.
    """
    scalars = jnp.array([0.0, 0.5, 1.0])
    expected = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    
    result = scalar_to_interval(scalars)
    assert jnp.array_equal(result, expected)
    assert result.shape == (3, 2)

def test_is_precise():
    """
    Tests the precision detection logic for truth intervals.
    
    An interval is considered 'precise' if its uncertainty width (U - L) 
    is near zero. This test checks both a perfectly collapsed interval 
    and a vague interval to ensure the threshold logic works correctly.
    """
    precise = jnp.array([0.5, 0.5])
    vague = jnp.array([0.1, 0.9])
    
    assert is_precise(precise) is True
    assert is_precise(vague) is False