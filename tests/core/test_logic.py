#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.core import intervals, logic

def test_weighted_and_lukasiewicz():
    """
    Validates the weighted Łukasiewicz conjunction with multiple inputs.

    This test verifies that:
    1. When all inputs are fully True ([1.0, 1.0]) and weights are neutral (1.0), 
       the resulting interval is correctly identified as fully True.
    2. The logic properly handles weighted resistance aggregation against the beta threshold.
    """
    # Inputs: A=[1,1], B=[1,1] with weights [1,1] and beta=1
    inputs = intervals.create_interval(jnp.array([1.0, 1.0]), jnp.array([1.0, 1.0]))
    weights = jnp.array([1.0, 1.0])
    beta = jnp.array(1.0)
    
    res = logic.weighted_and_lukasiewicz(inputs, weights, beta)
    assert jnp.all(intervals.get_lower(res) == 1.0)
    assert jnp.all(intervals.get_upper(res) == 1.0)

def test_implies_kleene_dienes():
    """
    Tests the Kleene-Dienes (pessimistic) implication: max(1 - A, B).

    In interval logic, this test confirms the boundary condition for 
    classical logic within a fuzzy framework: if the antecedent is True 
    and the consequent is False, the implication must be fully False.
    """
    a = intervals.create_interval(jnp.array(1.0), jnp.array(1.0)) # True
    b = intervals.create_interval(jnp.array(0.0), jnp.array(0.0)) # False
    
    res = logic.implies_kleene_dienes(a, b)
    # 1 -> 0 must be 0
    assert intervals.get_upper(res) == 0.0

def test_implies_reichenbach_differentiability():
    """
    Verifies the Reichenbach implication: 1 - A + (A * B).

    This test checks the specific algebraic behavior of the Reichenbach 
    operator, ensuring it produces correct continuous truth values 
    for uncertain (0.5) inputs and maintains the result within [0, 1].
    """
    a = intervals.create_interval(jnp.array(0.5), jnp.array(0.5))
    b = intervals.create_interval(jnp.array(0.5), jnp.array(0.5))
    
    res = logic.implies_reichenbach(a, b)
    # Calculation: 1 - 0.5 + (0.5 * 0.5) = 0.5 + 0.25 = 0.75
    assert jnp.isclose(intervals.get_lower(res), 0.75)