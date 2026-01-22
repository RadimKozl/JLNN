#!/usr/bin/env python3

# Imports
import jax
import jax.numpy as jnp
from flax import nnx
from jlnn.core import intervals

class LearnedPredicate(nnx.Module):
    """
    Maps scalar input data to truth intervals [L, U].

    This class acts as the input layer of a JLNN network (grounding). 
    Using trainable parameters (slope and offset), 
    it learns to define the semantics of logical statements over continuous data. 
    For example, it can learn what specific value of 
    a temperature sensor corresponds to the logical statement "it's hot".

    It uses two independent sigmoidal functions to model the lower (L) 
    and upper (U) bounds of truth, which allows the system to also express 
    the degree of uncertainty (ignorance) regarding a given feature.
    """
    def __init__(self, in_features: int, rngs: nnx.Rngs):
        """
        Initializes the predicate parameters for each input flag.

        Args:
            in_features (int): Number of input numeric features.
            rngs (nnx.Rngs): Random number generator for Flax NNX initialization.
        """
        # Parameters for the lower bound (L) – defines the confirmed truth.
        self.slope_l = nnx.Param(jnp.ones((in_features,)))
        self.bias_l = nnx.Param(jnp.zeros((in_features,)))
        
        # Parameters for upper bound (U) – defines the upper probability limit.
        self.slope_u = nnx.Param(jnp.ones((in_features,)))
        self.bias_u = nnx.Param(jnp.zeros((in_features,)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Transforms numeric inputs into logical intervals.

        Args:
            x (jnp.ndarray): Input data (features) of the form (..., in_features).

        Returns:
            jnp.ndarray: An interval tensor of the form (..., in_features, 2), 
                        where the last dimension represents the pair [L, U].
        """
        # Calculation using sigmoid (truncated linear functions are often used in LNNs,
        # but sigmoid provides smoother gradients for learning stability in JAX).
        lower = jax.nn.sigmoid(self.slope_l * x + self.bias_l)
        upper = jax.nn.sigmoid(self.slope_u * x + self.bias_u)
        
        # Ensuring logical consistency (L <= U).
        # If L > U, it would mean a logical conflict.
        real_upper = jnp.maximum(lower, upper)
        
        # Creating a standard JLNN interval
        return intervals.create_interval(lower, real_upper)