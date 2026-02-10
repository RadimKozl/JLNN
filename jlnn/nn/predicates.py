#!/usr/bin/env python3

# Imports
import jax
import jax.numpy as jnp
from flax import nnx
from jlnn.core import activations, intervals

class LearnedPredicate(nnx.Module):
    """
    Grounding layer that transforms real-valued input data into truth intervals [L, U].

    In Logical Neural Networks (LNN), predicates act as the interface between 
    numeric data and logical formulas. This class learns the semantic mapping 
    (grounding) by adjusting slopes and offsets of activation functions to 
    produce fuzzy truth values.

    Attributes:
        slope_l (nnx.Param): Steepness of the lower bound activation.
        offset_l (nnx.Param): Horizontal shift for the lower bound activation.
        slope_u (nnx.Param): Steepness of the upper bound activation.
        offset_u (nnx.Param): Horizontal shift for the upper bound activation.
    """
    def __init__(self, in_features: int, rngs: nnx.Rngs):
        """
        Initializes trainable parameters for each input feature.

        Args:
            in_features (int): Number of input features to be grounded.
            rngs (nnx.Rngs): Flax NNX random number generator collection.
        """
        # Slopes determine how quickly a value transitions from False to True.
        self.slope_l = nnx.Param(jnp.ones((in_features,)))
        self.offset_l = nnx.Param(jnp.zeros((in_features,)))
        
        # Offsets for U are initialized slightly differently to ensure initial L <= U.
        self.slope_u = nnx.Param(jnp.ones((in_features,)))
        self.offset_u = nnx.Param(jnp.full((in_features,), -0.2))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Maps numeric inputs to truth intervals using learned sigmoidal ramps.

        Args:
            x (jnp.ndarray): Numeric input tensor of shape (..., in_features).

        Returns:
            jnp.ndarray: Truth interval tensor of shape (..., in_features, 2).
        """
        # Note: Using Ellipsis [...] for parameter access is the modern Flax NNX 
        # standard to retrieve the underlying JAX Array without deprecation warnings.
        lower = activations.ramp_sigmoid(x, self.slope_l[...], self.offset_l[...])
        upper = activations.ramp_sigmoid(x, self.slope_u[...], self.offset_u[...])
        
        # Pack lower and upper bounds into a single interval representation [L, U].
        return intervals.create_interval(lower, upper)
      
class FixedPredicate(nnx.Module):
    """
    Non-trainable predicate that returns the input interval unchanged.

    This module serves as an identity transformation for truth intervals, 
    preserving the original lower and upper bounds. It is primarily utilized 
    for crisp boolean logic scenarios where fixed truth values are required 
    without neural weight updates.

    Attributes:
        None
    """
    def __init__(self):
        """
        Initializes the FixedPredicate module.
        """
        pass

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Passes the input interval through without modification.

        Args:
            x (jnp.ndarray): Input truth interval tensor of the form (..., 2) 
                representing [L, U].

        Returns:
            jnp.ndarray: The identical input interval [L, U].
        """
        return x  # identity – [L, U] returns [L, U]