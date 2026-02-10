#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from flax import nnx
from jlnn.nn import predicates
from jlnn.core import intervals

def test_learned_predicate_bounds(rngs):
    """
    Verifies that LearnedPredicate enforces logical domain and boundary invariants.

    This test checks the grounding layer's ability to map raw numeric data into
    the fuzzy logic domain [0.0, 1.0]. Specifically, it ensures that:
    1. **Invariant Enforcement**: The lower bound (L) never exceeds the upper bound (U),
       preventing nonsensical 'negative' interval widths.
    2. **Domain Saturation**: Extremely high or low input values are correctly
       saturated (clipped) within the [0.0, 1.0] range using the ramp activation.
    3. **Tensor Integrity**: The output maintains the expected interval shape (..., 2).

    Args:
        rngs (nnx.Rngs): Flax NNX random number generator collection.
    """
    n_feat = 2
    pred = predicates.LearnedPredicate(n_feat, rngs)
    
    # Test with extreme values to trigger saturation:
    # Large positive (expected ~1.0) and large negative (expected ~0.0)
    x = jnp.array([[10.0, -10.0]])
    res = pred(x)
    
    lower = intervals.get_lower(res)
    upper = intervals.get_upper(res)
    
    # 1. Consistency check: L <= U must hold for all grounded features
    assert jnp.all(lower <= upper), f"Boundary inversion detected: L > U in {res}"
    
    # 2. Domain check: Outputs must stay within logical bounds [0, 1]
    assert jnp.all(res >= 0.0) and jnp.all(res <= 1.0), f"Domain violation: {res} outside [0, 1]"
    
    # 3. Shape check: Output should be (batch, features, 2)
    assert res.shape == (1, 2, 2), f"Unexpected shape: {res.shape}"


def test_fixed_predicate_identity():
    """
    Validates that FixedPredicate acts as a transparent identity for truth intervals.

    FixedPredicates are essential for injecting 'hard' facts or pre-calculated
    groundings into the network. This test ensures that the module:
    1. Does not modify the input truth values (identity transformation).
    2. Correcty handles the interval dimension.
    """
    pred = predicates.FixedPredicate()
    
    # Input representing a known fuzzy fact [0.2, 0.8]
    x = intervals.create_interval(jnp.array(0.2), jnp.array(0.8))
    res = pred(x)
    
    assert jnp.array_equal(x, res), "FixedPredicate must return input unchanged."
    assert jnp.isclose(intervals.get_lower(res), 0.2)
    assert jnp.isclose(intervals.get_upper(res), 0.8)


def test_predicate_parameter_access(rngs):
    """
    Confirms correct parameter allocation and modern NNX access syntax.

    This test ensures that:
    1. Predicate parameters (slopes and offsets) are correctly initialized 
       as trainable nnx.Param objects.
    2. Parameters are accessible via the Ellipsis [...] syntax, ensuring 
       compatibility with the latest Flax NNX array-injection mechanisms.
    3. Initialization does not produce NaN or Inf values.

    Args:
        rngs (nnx.Rngs): Flax NNX random number generator collection.
    """
    n_feat = 3
    pred = predicates.LearnedPredicate(n_feat, rngs)
    
    # Check if parameters are reachable and have correct dimensionality
    assert pred.slope_l[...].shape == (n_feat,)
    assert pred.offset_u[...].shape == (n_feat,)
    
    # Ensure initialization is clean
    assert not jnp.any(jnp.isnan(pred.slope_l[...]))
    assert jnp.all(pred.slope_l[...] > 0), "Slopes should generally be positive on init."