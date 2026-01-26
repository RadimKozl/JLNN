#!/usr/bin/env python3

# Imports
import jax.numpy as jnp

def scalar_to_interval(x: jnp.ndarray) -> jnp.ndarray:
    """
    Converts standard [0, 1] probability scalars into precise [L, U] intervals.
    
    This is used to ground the JLNN model with data from classical datasets 
    where truth values are known exactly (L = U = x).
    
    Args:
        x (jnp.ndarray): Tensor of scalar truth values.
        
    Returns:
        jnp.ndarray: Tensor of intervals with shape (*x.shape, 2).
    """
    return jnp.stack([x, x], axis=-1)

def is_precise(interval: jnp.ndarray, epsilon: float = 1e-5) -> bool:
    """
    Checks if an interval has collapsed into a single point (L ≈ U).
    """
    return float(jnp.abs(interval[..., 0] - interval[..., 1])) < epsilon