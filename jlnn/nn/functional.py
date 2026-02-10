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
    Computes a weighted logical negation (NOT) with confidence scaling.

    This operation follows a two-stage process:
    1. **Pure Negation:** The input interval [L, U] is inverted to [1-U, 1-L] 
       preserving the uncertainty width.
    2. **Confidence Scaling:** The resulting negated interval is scaled by the 
       `weight` parameter. If weight > 1.0, the truth values are amplified 
       and then clipped to the [0, 1] domain.

    Args:
        x (jnp.ndarray): Input truth interval tensor of shape (..., 2).
        weight (jnp.ndarray): Scaling factor (0.0 to 1.0+) representing 
            the importance or confidence of the negation.

    Returns:
        jnp.ndarray: The scaled negated interval, clipped to [0, 1].
    """
    # 1. Pure negation: [L, U] -> [1-U, 1-L]
    negated = intervals.negate(x)
    
    # 2. Applying weight to the result of negation
    l_neg = intervals.get_lower(negated) * weight
    u_neg = intervals.get_upper(negated) * weight
    
    # 3. Merge and enforce domain [0, 1] and consistency L <= U
    combined = jnp.stack([l_neg, u_neg], axis=-1)
    # Explicit clip ensures that tests on domain bounds [0, 1] pass
    return intervals.ensure_interval(jnp.clip(combined, 0.0, 1.0))

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