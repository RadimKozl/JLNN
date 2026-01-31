#!/usr/bin/env python3

"""
Logical health and performance metrics for LNN models.
"""
# Imports
import jax.numpy as jnp
from jlnn.core import intervals

def contradiction_degree(interval: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the degree of logical contradiction within a truth interval.
    
    In LNN, a contradiction occurs when the lower bound (L) exceeds the 
    upper bound (U), meaning the evidence for truth is greater than the 
    evidence for possibility.
    
    Args:
        interval (jnp.ndarray): Truth interval tensor [L, U].
        
    Returns:
        jnp.ndarray: Magnitude of contradiction (max(0, L - U)).
    """
    l, u = intervals.get_lower(interval), intervals.get_upper(interval)
    return jnp.maximum(0.0, l - u)

def uncertainty_width(interval: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the width of the uncertainty gap (U - L).
    
    A width of 0.0 represents a precise truth value (classical logic), 
    while 1.0 represents complete ignorance (unknown).
    
    Args:
        interval (jnp.ndarray): Truth interval tensor [L, U].
        
    Returns:
        jnp.ndarray: The distance between upper and lower bounds.
    """
    l, u = intervals.get_lower(interval), intervals.get_upper(interval)
    return jnp.maximum(0.0, u - l)