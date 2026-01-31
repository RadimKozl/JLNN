#!/usr/bin/env python3
"""
Unit tests for Logical Neural Network (LNN) specific metrics.

These metrics are used to evaluate the logical consistency and 
the information state (certainty vs. ignorance) of the model's predictions.
"""

# Imports
import jax.numpy as jnp
from jlnn.utils.metrics import contradiction_degree, uncertainty_width

def test_contradiction_degree():
    """
    Ensures that the contradiction degree is only non-zero when the 
    logical axiom L <= U is violated.
    
    A contradiction occurs in an LNN when the lower bound (evidence for truth) 
    crosses the upper bound (evidence for possibility). This test verifies 
    that valid intervals yield 0.0, while invalid ones correctly quantify 
    the gap.
    """
    valid = jnp.array([0.2, 0.8])
    invalid = jnp.array([0.9, 0.1]) # L=0.9, U=0.1 -> violation of 0.8
    
    assert contradiction_degree(valid) == 0.0, "Valid interval should have zero contradiction."
    assert jnp.isclose(contradiction_degree(invalid), 0.8), "Failed to calculate correct contradiction magnitude."

def test_uncertainty_width():
    """
    Verifies the calculation of the uncertainty gap (U - L).
    
    The width represents the degree of ignorance. 
    - 0.0 indicates absolute certainty (classical true/false).
    - 1.0 indicates 'Unknown' (full range of possibility).
    This test ensures both ends of the spectrum are handled correctly.
    """
    precise = jnp.array([0.5, 0.5])
    unknown = jnp.array([0.0, 1.0])
    
    assert uncertainty_width(precise) == 0.0, "Precise interval should have zero width."
    assert uncertainty_width(unknown) == 1.0, "Unknown interval should have a width of 1.0."