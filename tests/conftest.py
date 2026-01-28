#!/usr/bin/env python3

import pytest
from flax import nnx
import jax

@pytest.fixture
def rngs():
    """
    Provides a centralized random number generator (RNG) for NNX modules.
    
    This fixture ensures that all model initializations within a single test 
    session use a predictable seed, allowing for deterministic and 
    reproducible test results.

    Returns:
        nnx.Rngs: An NNX-specific RNG container.
    """    
    return nnx.Rngs(42)

@pytest.fixture
def key():
    """
    Provides a standard JAX PRNG key for functional JAX operations.

    This fixture is useful for tests that require raw JAX random operations 
    (like generating input noise or shuffling data) outside the NNX framework.

    Returns:
        jax.Array: A JAX PRNG key (KeyArray).
    """    
    return jax.random.key(0)