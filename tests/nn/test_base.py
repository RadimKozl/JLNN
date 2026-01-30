#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.nn.base import LogicalElement

def test_logical_element_init(rngs):
    """
    Verifies the correct initialization of the LogicalElement base class.

    This test ensures that:
    1. The weights are initialized to a neutral value of 1.0 as per LNN 
       requirements.
    2. The parameter shapes correctly match the number of input channels.
    3. The beta threshold (bias) is correctly initialized to 1.0.
    4. Access to parameters is handled via the modern Flax NNX [...] syntax 
       to ensure JAX compatibility.

    Args:
        rngs (nnx.Rngs): Flax NNX random number generator collection.
    """
    n_inputs = 3
    element = LogicalElement(n_inputs, rngs)
    
    # Check weight shape and default values using [...] syntax
    assert element.weights[...].shape == (n_inputs,)
    assert jnp.all(element.weights[...] == 1.0)
    
    # Check beta threshold default value
    assert element.beta[...] == 1.0