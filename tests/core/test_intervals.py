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
    
def test_ensure_consistent():
    """
    Validates the boundary correction logic to prevent negative uncertainty widths.

    This test verifies that `ensure_consistent` correctly enforces the interval 
    invariant (L <= U). It specifically checks two critical scenarios:
    1.  **Identity Mapping:** Ensures that already valid intervals (where L <= U) 
        remain unchanged.
    2.  **Boundary Swapping:** Confirms that inverted intervals (where L > U), 
        often caused by weighted negations or continuous transformations, are 
        automatically corrected by swapping the bounds.

    This ensures that the uncertainty width (U - L) always remains non-negative, 
    preserving the mathematical integrity of the interval logic across the system.
    """
    # Case 1: Correct order (L <= U) - Should remain unchanged
    l_valid, u_valid = jnp.array(0.2), jnp.array(0.8)
    nl, nu = intervals.ensure_consistent(l_valid, u_valid)
    assert nl == 0.2 and nu == 0.8
    
    # Case 2: Inverted order (L > U) - Should be swapped to (0.5, 0.7)
    l_inv, u_inv = jnp.array(0.7), jnp.array(0.5)
    nl, nu = intervals.ensure_consistent(l_inv, u_inv)
    assert nl == 0.5 and nu == 0.7
    assert nl <= nu
    
def test_ensure_interval_tensor():
    """
    Validates vectorized interval correction across multi-dimensional tensors.

    This test ensures that `ensure_interval` correctly identifies and fixes 
    mathematical inconsistencies (L > U) within a batch of intervals without 
    altering valid data or tensor shapes. It verifies the robust handling of:
    1. Standard valid intervals (L < U).
    2. Inverted intervals (L > U) that require boundary swapping.
    3. Point intervals (L = U) representing absolute certainty.

    The test confirms that the operation is fully vectorized, which is essential 
    for maintaining high performance during large-scale Multi-Agent System (MAS) 
    simulations where thousands of logical states are updated simultaneously.

    Shape:
        Input: [Batch_Size, 2] -> Output: [Batch_Size, 2]
    """
    # Batch of intervals: [Valid, Inverted (L > U), Point Interval (L = U)]
    data = jnp.array([
        [0.1, 0.9], 
        [0.8, 0.2], 
        [0.4, 0.4]
    ])
    
    fixed = intervals.ensure_interval(data)
    
    # Check shape integrity (JAX transformations must preserve tensor geometry)
    assert fixed.shape == data.shape
    
    # Axiom: All lower bounds must be less than or equal to upper bounds
    assert jnp.all(fixed[..., 0] <= fixed[..., 1])
    
    # Specific verification: [0.8, 0.2] must be corrected to [0.2, 0.8]
    assert jnp.all(fixed[1] == jnp.array([0.2, 0.8]))
    
    # Verify that valid intervals remained untouched
    assert jnp.all(fixed[0] == jnp.array([0.1, 0.9]))

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
    Validates the interval negation axiom and boundary consistency.

    In interval-based fuzzy logic (LNN), the negation of an interval [L, U] 
    is formally defined as [1 - U, 1 - L]. This test ensures that:
    1. The transformation correctly maps truth values to their complements.
    2. The lower and upper bounds are correctly swapped during subtraction 
       to maintain mathematical integrity (L <= U).
    3. The operation remains consistent across different regions of the 
       [0, 1] truth space.

    Axiom: NOT([L, U]) = [1 - U, 1 - L]
    """
    l, u = jnp.array(0.2), jnp.array(0.8)
    interval = intervals.create_interval(l, u)
    negated = intervals.negate(interval)
    
    # Case 1: Standard logic: NOT [0.2, 0.8] -> [0.2, 0.8] in some fuzzy interpretations,
    # but strictly: [1-0.8, 1-0.2] = [0.2, 0.8]
    assert jnp.allclose(intervals.get_lower(negated), 0.2)
    assert jnp.allclose(intervals.get_upper(negated), 0.8)
    
    # Case 2: Shifted interval [0.1, 0.3] -> [1-0.3, 1-0.1] = [0.7, 0.9]
    l2, u2 = jnp.array(0.1), jnp.array(0.3)
    interval_2 = intervals.create_interval(l2, u2)
    negated_2 = intervals.negate(interval_2) # NOT [0.1, 0.3] -> [0.7, 0.9]
    
    assert jnp.allclose(intervals.get_lower(negated_2), 0.7)
    assert jnp.allclose(intervals.get_upper(negated_2), 0.9)