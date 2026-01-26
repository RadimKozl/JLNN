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
    return logic.weighted_and_lukasiewicz(x, weights, beta)

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
    return logic.weighted_or_lukasiewicz(x, weights, beta)

def weighted_not(x: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
    """
    Weighted negation of an interval.

    It first applies a weight to the interval boundaries (clipped to 1.0) 
    and then performs an inversion of the boundaries.

    Args:
        x (jnp.ndarray): Input interval [L, U].
        weight (jnp.ndarray): Scalar weight or tensor weight.

    Returns:
        jnp.ndarray: Negated interval [1-U_w, 1-L_w].
    """
    weighted_x = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(x) * weight),
        jnp.minimum(1.0, intervals.get_upper(x) * weight)
    )
    return intervals.negate(weighted_x)

def weighted_nand(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Weighted NAND calculated as the negation of weighted AND.
    """
    res_and = weighted_and(x, weights, beta)
    return intervals.negate(res_and)

def weighted_nor(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Weighted NOR calculated as the negation of the weighted OR.
    """
    res_or = weighted_or(x, weights, beta)
    return intervals.negate(res_or)

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
        return logic.implies_kleene_dienes(weighted_a, weighted_b)
    elif method == 'reichenbach':
        return logic.implies_reichenbach(weighted_a, weighted_b)
    else:
        raise ValueError(f"Method {method} is not supported.")