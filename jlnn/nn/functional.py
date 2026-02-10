#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.core import logic, intervals
from typing import Optional

def weighted_and(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Stateless weighted conjunction (AND) according to Łukasiewicz.

    Args:
        x (jnp.ndarray): Input interval tensor of the form (..., num_inputs, 2).
        weights (jnp.ndarray): Tensor weights for individual inputs.
        beta (jnp.ndarray): Gate sensitivity threshold.

    Returns:
        jnp.ndarray: The resulting truth interval [L, U].
    """
    # Calls low-level implementation from kernel
    results = logic.weighted_and_lukasiewicz(x, weights, beta)
    return intervals.ensure_interval(results)

def weighted_or(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Stateless weighted disjunction (OR) according to Łukasiewicz.

    Args:
        x (jnp.ndarray): Input interval tensor of the form (..., num_inputs, 2).
        weights (jnp.ndarray): Tensor weights for individual inputs.
        beta (jnp.ndarray): Gate sensitivity threshold.

    Returns:
        jnp.ndarray: The resulting truth interval [L, U].
    """
    # Calls low-level implementation from kernel
    results = logic.weighted_or_lukasiewicz(x, weights, beta)
    return intervals.ensure_interval(results)

def weighted_not(x: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the weighted logical negation (NOT) of a truth interval.

    In JLNN interval logic, negation is defined by the transformation:
    [L, U] -> [1 - U, 1 - L]. This function extends this principle by applying 
    a scaling weight to the input boundaries before the inversion.

    The operation ensures that:
    1. **Evidence Reversal:** Strong evidence for truth (high L) is converted into 
       strong evidence for falsehood (low U).
    2. **Weight Scaling:** Input importance is adjusted via the `weight` parameter, 
       with results strictly clipped to the [0, 1] logical domain.
    3. **Invariant Preservation:** The function explicitly prevents boundary inversion 
       (negative width) ensuring the output interval always satisfies L <= U.

    Args:
        x (jnp.ndarray): Input interval tensor of shape (..., 2).
        weight (jnp.ndarray): Scaling weight applied to the input signal.

    Returns:
        jnp.ndarray: The negated and consistent truth interval tensor.
    """
    # 1. Apply weight to both edges and crop to [0,1]
    l_w = jnp.minimum(1.0, intervals.get_lower(x) * weight)
    u_w = jnp.minimum(1.0, intervals.get_upper(x) * weight)

    # 2. Perform negation – boundaries are swapped (1-U becomes new L)
    new_lower = 1.0 - u_w
    new_upper = 1.0 - l_w

    # 3. Final safety check for numerical stability and consistency
    final_lower = jnp.minimum(new_lower, new_upper)
    final_upper = jnp.maximum(new_lower, new_upper)

    return intervals.create_interval(final_lower, final_upper)

def weighted_nand(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Weighted NAND calculated as the negation of weighted AND.
    """
    res_and = weighted_and(x, weights, beta)
    results = intervals.negate(res_and)
    return intervals.ensure_interval(results)

def weighted_nor(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Weighted NOR calculated as the negation of the weighted OR.
    """
    res_or = weighted_or(x, weights, beta)
    results = intervals.negate(res_or)
    return intervals.ensure_interval(results)

def weighted_implication(int_a: jnp.ndarray, int_b: jnp.ndarray, 
                         weights: jnp.ndarray, beta: jnp.ndarray, 
                         method: str = 'lukasiewicz') -> jnp.ndarray:
    """
    Functional calculation of logical implication (A -> B).

    It supports various semantics for modeling expert rules.

    Args:
        int_a (jnp.ndarray): Antecedent (presupposition).
        int_b (jnp.ndarray): Consequent (consequence).
        weights (jnp.ndarray): Weights for A and B.
        beta (jnp.ndarray): Gate Threshold.
        method (str): Method ('lukasiewicz', 'kleene_dienes', 'reichenbach').

    Returns:
        jnp.ndarray: The truth of the rule as an interval.
    """
    if method == 'lukasiewicz':
        return logic.implies_lukasiewicz(int_a, int_b, weights, beta)
    
    # Preprocessing weights for other methods
    weighted_a = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(int_a) * weights[0]),
        jnp.minimum(1.0, intervals.get_upper(int_a) * weights[0])
    )
    weighted_b = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(int_b) * weights[1]),
        jnp.minimum(1.0, intervals.get_upper(int_b) * weights[1])
    )

    if method == 'kleene_dienes':
        results = logic.implies_kleene_dienes(weighted_a, weighted_b)
    elif method == 'reichenbach':
        results = logic.implies_reichenbach(weighted_a, weighted_b)
    else:
        raise ValueError(f"Method {method} is not supported.")
    
    return intervals.ensure_interval(results)