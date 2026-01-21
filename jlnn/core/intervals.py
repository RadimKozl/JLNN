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
        interval (jnp.ndarray): Tensor intervalů o tvaru (..., 2). 
                            Poslední dimenze obsahuje dvojici [Lower Bound, Upper Bound].

    Returns:
        jnp.ndarray: A tensor containing the width of each interval of the shape (...). 
                    The resulting shape has one less dimension than the input tensor.
    """        
    
    # Calculation of the difference between the upper (U) and lower (L) limits.
    # We use the previously defined abstractions get_upper and get_lower.
   
    return get_upper(interval) - get_lower(interval)