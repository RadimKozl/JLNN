#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.nn import predicates

def test_learned_predicate_bounds(rngs):
    """
    Verifies that the LearnedPredicate produces valid truth intervals [L, U].

    This test checks the grounding layer's output to ensure:
    1. The lower bound (L) never exceeds the upper bound (U), maintaining 
       logical consistency.
    2. All resulting truth values are contained within the valid fuzzy 
       logic range [0.0, 1.0].
    
    Args:
        rngs (nnx.Rngs): Flax NNX random number generator collection.
    """
    n_feat = 1
    pred = predicates.LearnedPredicate(n_feat, rngs)
    
    # Test with a high input value to check saturation and consistency
    x = jnp.array([[10.0]])
    res = pred(x)
    
    lower = res[..., 0]
    upper = res[..., 1]
    
    # Consistency check: Lower bound must be <= Upper bound
    assert jnp.all(lower <= upper)
    # Range check: Outputs must be valid truth values in [0, 1]
    assert jnp.all(res >= 0.0) and jnp.all(res <= 1.0)

def test_predicate_parameter_access(rngs):
    """
    Confirms correct parameter allocation and modern NNX access syntax.

    This test ensures that:
    1. Predicate parameters (slopes/offsets) are correctly initialized 
       with the expected feature dimensions.
    2. Parameters are accessible using the modern [...] syntax, which is 
       required to avoid deprecation warnings in the latest Flax NNX versions.

    Args:
        rngs (nnx.Rngs): Flax NNX random number generator collection.
    """
    pred = predicates.LearnedPredicate(in_features=4, rngs=rngs)
    
    # Verify shape and access for slope parameters using the Ellipsis [...] operator
    assert pred.slope_l[...].shape == (4,)