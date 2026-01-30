#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.core import intervals

def test_create_and_get_bounds():
    """
    Verifies the creation of truth intervals and the extraction of their bounds.
    
    This test ensures that the interval representation correctly packs 
    lower (L) and upper (U) bounds and that the getter functions retrieve 
    the original values accurately.
    """
    l, u = jnp.array([0.2, 0.3]), jnp.array([0.8, 0.9])
    interval = intervals.create_interval(l, u)
    
    assert jnp.all(intervals.get_lower(interval) == l)
    assert jnp.all(intervals.get_upper(interval) == u)
    assert interval.shape == (2, 2)

def test_check_contradiction():
    """
    Verifies the detection of logical contradictions within an interval.
    
    A contradiction occurs in interval logic when the lower bound (L) 
    is strictly greater than the upper bound (U). This test ensures 
    that the system correctly identifies these invalid states.
    """
    valid = intervals.create_interval(jnp.array(0.2), jnp.array(0.8))
    invalid = intervals.create_interval(jnp.array(0.8), jnp.array(0.2))
    
    assert not intervals.check_contradiction(valid)
    assert intervals.check_contradiction(invalid)

def test_negate_interval():
    """
    Validates the interval negation axiom: NOT [L, U] = [1 - U, 1 - L].
    
    This test confirms that the negation operation correctly inverts 
    the truth interval while maintaining the proper lower and upper 
    bound relationship.
    """
    interval = intervals.create_interval(jnp.array(0.2), jnp.array(0.7))
    negated = intervals.negate(interval)
    
    assert jnp.isclose(intervals.get_lower(negated), 0.3)
    assert jnp.isclose(intervals.get_upper(negated), 0.8)