#!/usr/bin/env python3

# Imports 
import jax
import jax.numpy as jnp
from jlnn.core import intervals


def weighted_and_lukasiewicz(ints: jnp.ndarray, weights: jnp.ndarray, beta: float) -> jnp.ndarray:
    """
    Weighted AND operator (Łukasiewicz t-norm) according to IBM LNN specification.
    
    Calculation: 1 - min(1, sum(w_i * (1 - x_i)) / beta)
    
    Args:
        ints: Tensor of intervals of the form (..., N, 2), where N is the number of inputs.
        weights: Weights of individual inputs of the form (N,). Should be >= 1.
        beta: Threshold (bias) determining activation.
        
    Returns:
        The resulting interval [L, U] is of the form (..., 2).
    """
    L = intervals.get_lower(ints)
    U = intervals.get_upper(ints)
    
    # We use jnp.sum with axis=-1 so that the gate works for batches (multiple rows of data)
    res_L = 1.0 - jnp.minimum(1.0, jnp.sum(weights * (1.0 - L), axis=-1) / beta)
    res_U = 1.0 - jnp.minimum(1.0, jnp.sum(weights * (1.0 - U), axis=-1) / beta)
    
    return intervals.create_interval(res_L, res_U)


def weighted_or_lukasiewicz(ints: jnp.ndarray, weights: jnp.ndarray, beta: float) -> jnp.ndarray:
    """
    Weighted OR operator (Łukasiewicz t-conorma) for interval logic.
    
    This operator aggregates the truth intervals of several inputs into 
    a single output interval. Within LNN (Logical Neural Networks), 
    a linear form is used, which allows for efficient gradient propagation.
    
    The calculation is performed independently for the lower (L) 
    and upper (U) bounds according to the formula: 
    f(x) = min(1, sum(w_i * x_i) / beta)

    Args:
        ints (jnp.ndarray): Input tensor of intervals of the form (..., N, 2), 
                        where N is the number of operator arguments 
                        and the last dimension contains the pair [Lower Bound, Upper Bound].
        weights (jnp.ndarray): A tensor of weights of the form (N,). 
                        In accordance with LNN, weights should be >= 1 to ensure logical consistency.
        beta (float): The threshold (bias) of the gate. 
                        It determines the "sensitivity" of the OR gate - a lower beta means that 
                        fewer true inputs are needed to achieve full truth (1.0).

    Returns:
        jnp.ndarray: The resulting truth interval [L, U] of the form (..., 2).
    """
    
    # Extracting boundaries using our intervals module
    L = intervals.get_lower(ints)
    U = intervals.get_upper(ints)
    
    # Calculate the weighted sum over the last dimension of the arguments (axis=-1)
    # This provides support for an arbitrary number of batch dimensions.
    res_L = jnp.minimum(1.0, jnp.sum(weights * L, axis=-1) / beta)
    res_U = jnp.minimum(1.0, jnp.sum(weights * U, axis=-1) / beta)
    
    # Reconstruction to format (..., 2)
    return intervals.create_interval(res_L, res_U)


def implies_lukasiewicz(int_a: jnp.ndarray, int_b: jnp.ndarray, weights: jnp.ndarray, beta: float) -> jnp.ndarray:
    """
    Logical implication A -> B (S-implication) based on Łukasiewicz logic.
    
    In JLNN, implication is implemented using logical equivalence:
    (A -> B) ≡ (¬A ∨ B).
    
    This implementation uses interval arithmetic, 
    where negation (NOT) inverts the interval boundaries: NOT [L, U] = [1 - U, 1 - L]. 
    The result is then processed by a weighted OR operator, 
    allowing the model to learn the relevance of a given rule.

    Args:
        int_a (jnp.ndarray): Tensor for the antecedent (presupposition A) of the form (..., 2). 
                            The last dimension contains [Lower Bound, Upper Bound].
        int_b (jnp.ndarray): Tensor for the consequent (consequent B) of the form (..., 2). 
                            The last dimension contains [Lower Bound, Upper Bound].
        weights (jnp.ndarray): Tensor of weights for an OR gate of the form (2,). 
                            The first weight is applied to ¬A, the second to B. Typically initialized to [1, 1].
        beta (float): Threshold (bias) determining the stringency of the implication activation.

    Returns:
        jnp.ndarray: The resulting truth interval of the implication [L, U] of the form (..., 2).
    """
    
    
    # NOT in interval logic: [1-U, 1-L]
    # We use helper functions from our intervals module for code cleanliness
    not_a_L = 1.0 - intervals.get_upper(int_a)
    not_a_U = 1.0 - intervals.get_lower(int_a)
    not_a = intervals.create_interval(not_a_L, not_a_U)
    
    # Group NOT A and B into one tensor for bulk processing in an OR gate.
    # axis=-2 creates a dimension for the arguments before the last (interval) dimension.
    combined = jnp.stack([not_a, int_b], axis=-2)
    
    # Call the previously defined weighted OR
    return weighted_or_lukasiewicz(combined, weights, beta)
