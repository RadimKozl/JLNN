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


class PhysicalPredicate(nnx.Module):
    """
    Grounding layer using Space-Curved Physical Fuzzy Logic (PFL).
    
    Transforms real-valued input data into truth intervals [L, U] using 
    gravitational space deformation. Unstable and highly uncertain states 
    are naturally pulled toward the entropic singularity (0.5), while deterministic 
    edges saturate safely.
    
    Attributes:
        slope_l (nnx.Param): Steepness of the lower bound potential.
        offset_l (nnx.Param): Center shift for the lower bound potential.
        slope_u (nnx.Param): Steepness of the upper bound potential.
        offset_u (nnx.Param): Center shift for the upper bound potential.
        gamma (float): Strength of the gravitational bending towards 0.5.
        mode (str): Base compression method ('sigmoid' or 'ramp').
    """
    def __init__(self, in_features: int, rngs: nnx.Rngs, gamma: float = 0.2, mode: str = 'sigmoid'):
        """
        Initializes trainable parameters and PFL parameters.
        """
        self.gamma = gamma
        self.mode = mode

        # Trainable parameters for potential landscape mapping
        self.slope_l = nnx.Param(jnp.ones((in_features,)))
        self.offset_l = nnx.Param(jnp.zeros((in_features,)))
        
        # Offset for upper bound is slightly shifted to encourage initial L <= U constraint
        self.slope_u = nnx.Param(jnp.ones((in_features,)))
        self.offset_u = nnx.Param(jnp.full((in_features,), -0.2))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Maps numeric inputs to physical truth intervals using gravitational deformation.
        """
        # 1. Compute input potentials (z) for lower and upper bounds
        # (Standard linear mapping before entering the curved space activation)
        z_l = self.slope_l[...] * (x - self.offset_l[...])
        z_u = self.slope_u[...] * (x - self.offset_u[...])

        # 2. Pass potentials through the physical gravitational activation function
        # For mode='ramp', we supply the internal slope=1.0 and offset=0.5 because 
        # the parameters were already applied during potential (z) construction.
        lower = activations.gravitational_bend_activation(
            z_l, gamma=self.gamma, mode=self.mode, slope=1.0, offset=0.5
        )
        upper = activations.gravitational_bend_activation(
            z_u, gamma=self.gamma, mode=self.mode, slope=1.0, offset=0.5
        )
        
        # 3. Enforce valid interval logic constraints [L <= U] and pack
        # (Using safe clip/minimum check inside ensure_interval if needed, or direct creation)
        combined = intervals.create_interval(lower, upper)
        return intervals.ensure_interval(combined)
    

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