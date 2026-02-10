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

    FIXED VERSION - Corrects the weighting semantics to preserve NOT logic.

    The weight parameter controls how much the negation "pulls toward uncertainty":
    - weight = 1.0: Pure logical NOT, [L, U] -> [1-U, 1-L]
    - weight < 1.0: Negation is "softened" by blending with the unknown state [0, 1]
    - weight > 1.0: Not recommended (would push values outside [0, 1])

    The weighting is applied by linear interpolation:
        result = weight * NOT(x) + (1 - weight) * [0, 1]

    This ensures that when weight = 1.0, we get pure logical negation.
    When weight = 0.0, we get complete uncertainty [0, 1].

    Args:
        x (jnp.ndarray): Input truth interval tensor of shape (..., 2).
        weight (jnp.ndarray): Scaling factor (0.0 to 1.0) representing 
            the confidence or strength of the negation.

    Returns:
        jnp.ndarray: The weighted negated interval, clipped to [0, 1].

    Examples:
        >>> # Pure negation (weight = 1.0)
        >>> x = jnp.array([[0.0, 0.0]])  # False
        >>> weighted_not(x, jnp.array(1.0))
        # Returns: [[1.0, 1.0]] (True) ✓
        
        >>> x = jnp.array([[1.0, 1.0]])  # True
        >>> weighted_not(x, jnp.array(1.0))
        # Returns: [[0.0, 0.0]] (False) ✓
        
        >>> # Softened negation (weight = 0.5)
        >>> x = jnp.array([[0.0, 0.0]])  # False
        >>> weighted_not(x, jnp.array(0.5))
        # Returns: [[0.5, 1.0]] (blend of True and Unknown)
    """
    # 1. Pure negation: [L, U] -> [1-U, 1-L]
    negated = intervals.negate(x)
    
    # 2. Linear interpolation between negated result and maximum uncertainty [0, 1]
    # When weight = 1.0: result = negated (pure NOT)
    # When weight = 0.0: result = [0, 1] (complete ignorance)
    l_neg = intervals.get_lower(negated) * weight + 0.0 * (1.0 - weight)
    u_neg = intervals.get_upper(negated) * weight + 1.0 * (1.0 - weight)
    
    # 3. Merge and enforce domain [0, 1] and consistency L <= U
    combined = jnp.stack([l_neg, u_neg], axis=-1)
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