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
    2. Saturation logic: extremely high or low values correctly saturate at \
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
    
    In Logical Neural Networks, the AND operation is governed by a 'resistance' \
    sum (sum_val) and a threshold (beta). This test ensures:
    1. If the sum of resistance is zero, the output is fully True (1.0).
    2. If the sum exceeds the beta threshold, the output is fully False (0.0).
    """
    # Case 1: Minimal resistance (0.0), output should be absolute truth (1.0)
    assert activations.lukasiewicz_and_activation(sum_val=0.0, beta=1.0) == 1.0
    
    # Case 2: Resistance exactly hits the beta threshold, output collapses to 0.0
    assert activations.lukasiewicz_and_activation(sum_val=1.0, beta=1.0) == 0.0
    
    # Case 3: Resistance exceeds threshold, hard saturation at 0.0
    assert activations.lukasiewicz_and_activation(sum_val=2.5, beta=1.0) == 0.0

def test_lukasiewicz_or_activation():
    """
    Verifies the threshold-based activation logic for the Łukasiewicz disjunction.
    
    The OR operation evaluates supportive evidence (sum_val) against a sensitivity \
    threshold (beta). This test ensures:
    1. Zero support yields absolute falsehood (0.0).
    2. Support reaching or exceeding beta triggers full saturation at absolute truth (1.0).
    """
    # Case 1: No support (0.0), output should be absolute falsehood (0.0)
    assert activations.lukasiewicz_or_activation(sum_val=0.0, beta=1.0) == 0.0
    
    # Case 2: Support exactly matches beta, output reaches 1.0
    assert activations.lukasiewicz_or_activation(sum_val=1.0, beta=1.0) == 1.0
    
    # Case 3: Overflowing support, hard saturation at 1.0
    assert activations.lukasiewicz_or_activation(sum_val=5.0, beta=1.0) == 1.0


# =====================================================================
# NEW: ENTROPIC AND PHYSICAL FUZZY LOGIC ACTIVATION TESTS
# =====================================================================

def test_entropy_raw_boundaries():
    """Verifies that the calculation of Shannon entropy converges correctly at the edges and at the maximum."""
    # At the point 0.5, entropy is maximal (equal to 1.0)
    assert jnp.allclose(activations.entropy_raw(jnp.array(0.5)), 1.0)
    
    # At the edges (0.0 and 1.0), entropy should converge to 0.0 (absolute certainty)
    assert jnp.allclose(activations.entropy_raw(jnp.array(0.0)), 0.0, atol=1e-6)
    assert jnp.allclose(activations.entropy_raw(jnp.array(1.0)), 0.0, atol=1e-6)


def test_get_entropic_weight():
    """Verifies that the entropic stability weight (1 - H) is correctly mapped."""
    # At maximal uncertainty (0.5), stability weight must collapse to 0.0
    assert jnp.allclose(activations.get_entropic_weight(jnp.array(0.5)), 0.0, atol=1e-6)
    
    # At deterministic boundaries, stability weight must reach 1.0
    assert jnp.allclose(activations.get_entropic_weight(jnp.array(0.0)), 1.0, atol=1e-6)
    assert jnp.allclose(activations.get_entropic_weight(jnp.array(1.0)), 1.0, atol=1e-6)


def test_gravitational_bend_activation_modes():
    """Verifies that the entropic deformation of space works stably for different modes and parameters."""
    
    # 1. Test for the default mode 'sigmoid' (midpoint equilibrium is at z = 0.0)
    z_sigmoid = jnp.array([-2.0, 0.0, 2.0])
    res_sigmoid = activations.gravitational_bend_activation(z_sigmoid, gamma=0.2, mode='sigmoid')
    assert res_sigmoid.shape == z_sigmoid.shape
    assert jnp.all(res_sigmoid >= 0.0) and jnp.all(res_sigmoid <= 1.0)
    
    # At the symmetric zero potential z=0, the center must remain exactly at 0.5 (gravitational equilibrium)
    assert jnp.allclose(res_sigmoid[1], 0.5)

    # 2. Test for the 'ramp' mode
    # For ramp, the midpoint equilibrium (0.5) occurs exactly when z == offset.
    offset_val = 0.5
    z_ramp = jnp.array([-1.0, offset_val, 2.0])
    res_ramp = activations.gravitational_bend_activation(z_ramp, gamma=0.1, mode='ramp', slope=1.0, offset=offset_val)
    assert res_ramp.shape == z_ramp.shape
    assert jnp.all(res_ramp >= 0.0) and jnp.all(res_ramp <= 1.0)
    
    # When z equals offset, the base_truth is 0.5, yielding maximal entropy and staying at 0.5
    assert jnp.allclose(res_ramp[1], 0.5)


def test_gravitational_bend_activation_invalid_mode():
    """Verifies that the entropic activation correctly raises a ValueError for an unknown mode."""
    with pytest.raises(ValueError, match="Unknown PFL activation mode"):
        activations.gravitational_bend_activation(jnp.array([0.5]), mode='invalid_mode')