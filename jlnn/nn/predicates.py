#!/usr/bin/env python3

# Imports
import jax
import jax.numpy as jnp
from flax import nnx
from jlnn.core import activations, intervals

class LearnedPredicate(nnx.Module):
    """
    Transforms real input data into truth intervals [L, U].

    This class serves as the input layer of the JLNN network (so-called grounding). 
    Using trainable parameters (slope and offset), 
    it learns to define the semantics of logical statements over continuous data. For example, 
    it can learn what specific value from a temperature sensor corresponds to the logical statement "it's hot".

    It uses two independent ramp_sigmoid functions to model the lower (L) 
    and upper (U) bounds of truth, which allows the system to also express 
    the degree of uncertainty (ignorance) regarding a given feature.
    """
    def __init__(self, in_features: int, rngs: nnx.Rngs):
        """
        Initializes the predicate parameters for each input flag.

        Args:
            in_features (int): Number of numeric input features.
            rngs (nnx.Rngs): Random number generator for Flax NNX.
        """
        # Parameters for the lower limit (L) – defines the confirmed truth.
        self.slope_l = nnx.Param(jnp.ones((in_features,)))
        self.offset_l = nnx.Param(jnp.zeros((in_features,)))
        
        # Parameters for upper bound (U) – defines the upper limit of the probability.
        # We initialize offset_u slightly lower than offset_l so that U > L at the beginning.
        self.slope_u = nnx.Param(jnp.ones((in_features,)))
        self.offset_u = nnx.Param(jnp.full((in_features,), -0.2))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Transforms numeric inputs into logical intervals.

        Args:
            x (jnp.ndarray): Input data of the shape (..., in_features).

        Returns:
            jnp.ndarray: An interval tensor of the form (..., in_features, 2), 
                        where the last dimension represents the pair [L, U].
        """
        # Calculation using ramp_sigmoid from core/activations.py
        # This function provides a linear transition and saturation at 0.0 and 1.0.
        lower = activations.ramp_sigmoid(x, self.slope_l.value, self.offset_l.value)
        upper = activations.ramp_sigmoid(x, self.slope_u.value, self.offset_u.value)
        
        # In JLNN, we do not perform jnp.maximum(lower, upper) inside the forward pass.
        # We want any conflict (L > U) to be visible to contradiction_loss.
        # The correction is handled by apply_constraints after each optimization step.
        
        return intervals.create_interval(lower, upper)