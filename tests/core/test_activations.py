#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
import pytest
from jlnn.core import activations

def test_identity_activation():
    """
    Verifies that the identity activation correctly clamps values to the [0, 1] range.
    
    This test ensures that any input falling outside the logical bounds of fuzzy 
    logic (below 0.0 or above 1.0) is properly trimmed to maintain semantic 
    validity within the network.
    """
    x = jnp.array([-0.5, 0.0, 0.5, 1.0, 1.5])
    expected = jnp.array([0.0, 0.0, 0.5, 1.0, 1.0])
    res = activations.identity_activation(x)
    assert jnp.allclose(res, expected)

def test_ramp_sigmoid():
    """
    Tests the linear transformation of real numbers into truth values using the ramp sigmoid.
    
    This test confirms:
    1. The midpoint behavior: where input equals the offset, the truth value is 0.5.
    2. Saturation logic: extremely high or low values correctly saturate at 
       exactly 1.0 (True) or 0.0 (False).
    """
    # The midpoint (offset) must correspond to a truth value of 0.5
    assert activations.ramp_sigmoid(0.5, slope=1.0, offset=0.5) == 0.5
    # Saturation test
    assert activations.ramp_sigmoid(10.0, slope=1.0, offset=0.5) == 1.0
    assert activations.ramp_sigmoid(-10.0, slope=1.0, offset=0.5) == 0.0

def test_lukasiewicz_and_activation():
    """
    Verifies the threshold-based activation logic for the Łukasiewicz conjunction.
    
    In Logical Neural Networks, the AND operation is governed by a 'resistance' 
    sum (s) and a threshold (beta). This test ensures:
    1. If the sum of resistance is zero, the output is fully True (1.0).
    2. If the sum exceeds the beta threshold, the output is fully False (0.0).
    """
    # s = 0 (no resistance) -> result 1.0
    assert activations.lukasiewicz_and_activation(jnp.array(0.0), jnp.array(1.0)) == 1.0
    # s >= beta (resistance exceeded threshold) -> result 0.0
    assert activations.lukasiewicz_and_activation(jnp.array(1.5), jnp.array(1.0)) == 0.0