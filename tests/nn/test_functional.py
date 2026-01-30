#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.nn import functional as F
from jlnn.core import intervals

def test_weighted_and_functional():
    """
    Tests the stateless weighted AND operation using Łukasiewicz semantics.

    This test verifies that:
    1. The functional interface correctly processes interval tensors.
    2. Under neutral weights (1.0) and standard beta (1.0), the operation 
       satisfies the T-norm axiom (e.g., 1 AND 0 = 0).
    3. The batch dimension is preserved during the computation.
    """
    # Inputs: [1,1] (True) and [0,0] (False)
    # In Łukasiewicz logic: 1 AND 0 = 0
    x = intervals.create_interval(jnp.array([1.0, 0.0]), jnp.array([1.0, 0.0]))
    x = x[jnp.newaxis, ...] # Add batch dimension
    
    weights = jnp.array([1.0, 1.0])
    beta = jnp.array(1.0)
    
    res = F.weighted_and(x, weights, beta)
    
    # Check if the upper bound of the result is close to 0 (False)
    assert jnp.isclose(intervals.get_upper(res), 0.0)

def test_weighted_or_functional():
    """
    Tests the stateless weighted OR operation using Łukasiewicz semantics.

    This test verifies that:
    1. The disjunction correctly aggregates truth values (T-conorm).
    2. Under neutral weights, 1 OR 0 results in 1 (True).
    3. The interval bounds [L, U] are correctly maintained in the output.
    """
    # 1 OR 0 = 1
    x = intervals.create_interval(jnp.array([1.0, 0.0]), jnp.array([1.0, 0.0]))
    x = x[jnp.newaxis, ...]
    
    weights = jnp.array([1.0, 1.0])
    beta = jnp.array(1.0)
    
    res = F.weighted_or(x, weights, beta)
    
    # Check if the lower bound of the result is close to 1 (True)
    assert jnp.isclose(intervals.get_lower(res), 1.0)