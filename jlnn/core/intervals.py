#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from typing import Tuple


def create_interval(lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
    """
    Creates an interval tensor representing truth values as [Lower Bound, Upper Bound].
    
    This function groups two separate tensors (lower and upper bounds) into one common tensor. 
    In the JLNN architecture, this format is the standard for 
    representing uncertainty - the greater the difference between the upper and lower bounds, 
    the greater the system's ignorance about the given fact.

    Args:
        lower (jnp.ndarray): Tensor containing lower truth bounds (0.0 to 1.0). Can have any shape.
        upper (jnp.ndarray): Tensor containing upper truth limits (0.0 to 1.0). 
            Must have the same shape as the 'lower' parameter.

    Returns:
        jnp.ndarray: The resulting interval tensor of the form (..., 2), 
            where the last dimension contains the pair [L, U]. 
            This format is optimized for efficient use in JAX transformations such as vmap and jit.
    """
    return jnp.stack([lower, upper], axis=-1)


def get_lower(interval: jnp.ndarray) -> jnp.ndarray:
    """
    Extracts the lower bound from an interval tensor.
    
    In the JLNN architecture, the lower bound represents 
    the minimum degree of confirmed truth of a given statement. 
    If L=1, the statement is considered provably true.

    Args:
        interval (jnp.ndarray): A tensor of intervals of the form (..., 2). 
            The last dimension is assumed to contain the pair [L, U].

    Returns:
        jnp.ndarray: A tensor containing only the lower bounds (L). 
            The resulting shape is (...,), one dimension less than the input.
    """
    
    # Using the ellipsis (...) ensures that the function works for any
    # number of batch dimensions and always selects the first element from the last pair.
    return interval[..., 0]

def get_upper(interval: jnp.ndarray) -> jnp.ndarray:
    """
    Extracts the upper bound (Upper Bound) from the interval tensor.
    
    In LNN logic, the upper bound (U) represents the maximum possible degree of truth 
    that a given statement can have given the absence of evidence of its falsity. 
    If U=0, the statement is provably false (False).

    Args:
        interval (jnp.ndarray): A tensor of intervals of the form (..., 2). 
            The last dimension is assumed to contain the pair [L, U].

    Returns:
        jnp.ndarray: A tensor containing only upper bounds (U). 
            The resulting shape is (...,), one dimension less than the input tensor.
    """    
    # Index 1 selects the Upper Bound from the last dimension.
    # The ellipsis (...) ensures compatibility with any number of batch dimensions.
    return interval[..., 1]


def check_contradiction(interval: jnp.ndarray) -> jnp.ndarray:
    """
    Detects a logical contradiction in the truth interval.
    
    Within LNN (Logical Neural Networks), we work with intervals [L, U], 
    where L (Lower Bound) is the level of evidence for truth 
    and U (Upper Bound) is the upper limit above which there is no more evidence for falsity.
    
    A logical contradiction occurs when L > U. This means that the sum of evidence for truth 
    and falsehood is so high that the boundaries have been "crossed." 
    This condition indicates inconsistent learning or conflicting data in the knowledge base.


    Args:
        interval (jnp.ndarray): Input tensor of intervals of the form (..., 2). 
            The last dimension is assumed to contain the pair [Lower Bound, Upper Bound].

    Returns:
        jnp.ndarray: Boolean tensor (or float tensor 1.0/0.0 in JAX) of the form (...). 
            A value of True (1.0) indicates that a conflict occurred in the given interval (L > U).
    """    
    
    # We use previously defined helper functions for extracting limits.
    # jnp.greater performs element-wise comparison across all batch dimensions.
    return jnp.greater(get_lower(interval), get_upper(interval))


def uncertainty(interval: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the uncertainty as the width of the interval (U - L).
    
    In Logical Neural Networks (LNN), the width of the interval represents 
    the so-called epistemic uncertainty - that is, 
    how much information we lack to be able to decide on a given statement with absolute certainty.
    
    Meaning of the resulting values:
        - 1.0: Complete ignorance (interval [0, 1]). We have no evidence for the statement.
        - 0.0: Absolute certainty (e.g. [1, 1] for True or [0, 0] for False).
        - Values close to 0: Signal that the system is approaching a specific conclusion.
        - Negative values: Indicate a logical contradiction (L > U).

    Args:
        interval (jnp.ndarray): A tensor of intervals of the form (..., 2). 
            The last dimension contains the pair [Lower Bound, Upper Bound].

    Returns:
        jnp.ndarray: A tensor containing the width of each interval of the shape (...). 
            The resulting shape has one less dimension than the input tensor.
    """        
    
    # Calculation of the difference between the upper (U) and lower (L) limits.
    # We use the previously defined abstractions get_upper and get_lower.
   
    return get_upper(interval) - get_lower(interval)


def negate(interval: jnp.ndarray) -> jnp.ndarray:
    """
    Performs a logical negation (NOT) over a truth interval.
    
    In interval logic JLNN, negation is defined by the relation:
        NOT [L, U] = [1 - U, 1 - L].
        
    This calculation ensures that:
        1. What was strong evidence for truth (high L) becomes strong evidence for falsehood (low U).
        2. The degree of ignorance (interval width) remains unchanged.

    Args:
        interval (jnp.ndarray): Input tensor of intervals of the form (..., 2). 
            The last dimension contains the pair [Lower Bound, Upper Bound].

    Returns:
        jnp.ndarray: Negated interval tensor of form (..., 2).
    """    
   
    # Extracting boundaries using previously defined getters
    lower = get_lower(interval)
    upper = get_upper(interval)
    
    # Calculation of negation: lower limit of result is 1 - upper limit of input and vice versa
    return create_interval(1.0 - upper, 1.0 - lower)


def ensure_consistent(lower: jnp.ndarray, upper: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Enforces interval consistency by ensuring that the lower bound is less than or equal to the upper bound.

    In neuro-symbolic and fuzzy logic operations, continuous transformations (like weighted 
    negations) can sometimes invert interval boundaries. This function corrects such 
    inversions using vectorized minimum and maximum operations, maintaining compatibility 
    with JAX's JIT compilation and auto-differentiation.

    Args:
        lower: A JAX array representing the lower bounds (L) of the intervals.
        upper: A JAX array representing the upper bounds (U) of the intervals.

    Returns:
        A tuple of (new_lower, new_upper) where new_lower <= new_upper for all elements.
        The shapes of the returned arrays match the input shapes.

    Example:
        >>> L, U = jnp.array(0.7), jnp.array(0.5)
        >>> ensure_consistent(L, U)
        (DeviceArray(0.5, dtype=float32), DeviceArray(0.7, dtype=float32))
    """
    # Vectorized swap logic – works for scalars, batches, and high-dimensional tensors
    new_lower = jnp.minimum(lower, upper)
    new_upper = jnp.maximum(lower, upper)
    return new_lower, new_upper


def ensure_interval(array: jnp.ndarray) -> jnp.ndarray:
    """
    Validates and corrects an entire interval tensor to ensure mathematical consistency.

    This function processes a tensor where the last dimension represents an interval 
    [lower_bound, upper_bound]. It decomposes the tensor, ensures that each lower 
    bound is less than or equal to its corresponding upper bound using 
    `ensure_consistent`, and reconstructs the tensor.

    This is a critical safety utility for neuro-symbolic layers where batch 
    operations or weighted logic transformations might invert interval limits.

    Args:
        array: A JAX array of shape [..., 2], where the last axis contains 
               the interval pairs.

    Returns:
        A JAX array of the same shape [..., 2] with corrected (sorted) intervals 
        along the last axis.

    Example:
        >>> data = jnp.array([[0.8, 0.2], [0.1, 0.9]])
        >>> ensure_interval(data)
        DeviceArray([[0.2, 0.8], [0.1, 0.9]], dtype=float32)
    """
    lower = array[..., 0]
    upper = array[..., 1]
    new_lower, new_upper = ensure_consistent(lower, upper)
    return jnp.stack([new_lower, new_upper], axis=-1)