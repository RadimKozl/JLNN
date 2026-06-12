#!/usr/bin/env python3

# Imports
import jax
import jax.numpy as jnp
from flax import nnx
from jlnn.core import activations, intervals

class LearnedPredicate(nnx.Module):
    """
    Stateful parametric grounding layer that transforms real-valued input data 
    into bounded truth intervals [L, U].

    In Logical Neural Networks (LNN), predicates function as the foundational semantic 
    interface mapping raw empirical data streams into strict logical propositions. 
    This class models parametric grounding profiles by independently optimizing the slopes 
    and offsets of monotonic activation functions, thereby refining fuzzy boundaries 
    via backpropagation while preserving structural valid intervals.

    Attributes:
        slope_l (nnx.Param): Trainable sensitivity vector scaling the lower truth bound activation.
        offset_l (nnx.Param): Trainable horizontal threshold vector shifting the lower truth bound.
        slope_u (nnx.Param): Trainable sensitivity vector scaling the upper truth bound activation.
        offset_u (nnx.Param): Trainable horizontal threshold vector shifting the upper truth bound.
    """
    def __init__(self, in_features: int, rngs: nnx.Rngs):
        """
        Initializes trainable slope and offset tracking parameters for each feature sub-space.

        Args:
            in_features (int): Total dimensionality of the incoming numerical feature vector.
            rngs (nnx.Rngs): Flax NNX random number generator collection for parameter states.
        """
        # Slopes control the gradient steepness of the truth transition region (False to True).
        self.slope_l = nnx.Param(jnp.ones((in_features,)))
        self.offset_l = nnx.Param(jnp.zeros((in_features,)))
        
        # Offsets for the upper bound (U) are purposefully initialized with a negative
        # translation bias to mathematically guarantee the fundamental L <= U condition at inception.
        self.slope_u = nnx.Param(jnp.ones((in_features,)))
        self.offset_u = nnx.Param(jnp.full((in_features,), -0.2))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Maps standard numerical arrays into multi-dimensional fuzzy truth intervals.

        Args:
            x (jnp.ndarray): Numerical observation tensor structured as (..., in_features).

        Returns:
            jnp.ndarray: Evaluated truth interval tensor structured as (..., in_features, 2), 
                where the trailing dimension defines the [Lower, Upper] truth limits.
        """
        # Note: Accessing parameters via the Ellipsis [...] index operator is the standard 
        # practice in Flax NNX to unpack raw JAX Arrays without raising future deprecation alerts.
        lower = activations.ramp_sigmoid(x, self.slope_l[...], self.offset_l[...])
        upper = activations.ramp_sigmoid(x, self.slope_u[...], self.offset_u[...])
        
        # Pack lower and upper bounds into a single interval representation [L, U].
        return intervals.create_interval(lower, upper)


class PhysicalPredicate(nnx.Module):
    """
    Stateful non-Euclidean grounding layer utilizing Space-Curved Physical Fuzzy Logic (PFL).
    
    Transforms real-valued external arrays into coherent logical truth intervals [L, U] 
    by routing numerical potentials through an artificial gravitational space deformation field. 
    Under this paradigm, highly unstable or contradictory information matrices are naturally 
    drawn towards the entropic singularity core (0.5), while highly deterministic inputs 
    converge safely into saturated axiomatic boundaries.
    
    Attributes:
        gamma (float): Coupling constant governing the intensity of the gravitational 
            restoring force pulling states towards absolute maximum entropy (0.5).
        mode (str): Baseline structural activation compression kernel ('sigmoid' or 'ramp').
        slope_l (nnx.Param): Trainable directional landscape steepness for lower bound potential.
        offset_l (nnx.Param): Trainable landscape origin shift for lower bound potential.
        slope_u (nnx.Param): Trainable directional landscape steepness for upper bound potential.
        offset_u (nnx.Param): Trainable landscape origin shift for upper bound potential.
    """
    def __init__(self, in_features: int, rngs: nnx.Rngs, gamma: float = 0.2, mode: str = 'sigmoid'):
        """
        Initializes trainable metric mapping landscapes alongside space curvature fields.

        Args:
            in_features (int): Total dimensionality of the incoming numerical feature vector.
            rngs (nnx.Rngs): Flax NNX random number generator collection.
            gamma (float, optional): Space bending elasticity coefficient. Defaults to 0.2.
            mode (str, optional): Underlying kernel geometry selector. Defaults to 'sigmoid'.
        """
        self.gamma = gamma
        self.mode = mode

        # Optimization states mapping input metrics to a standardized logical potential space
        self.slope_l = nnx.Param(jnp.ones((in_features,)))
        self.offset_l = nnx.Param(jnp.zeros((in_features,)))
        
        # Upper potential boundary initialization incorporates an axiomatic safety shift
        self.slope_u = nnx.Param(jnp.ones((in_features,)))
        self.offset_u = nnx.Param(jnp.full((in_features,), -0.2))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Maps numeric input arrays to curved physical truth intervals via gravitational warping.

        Args:
            x (jnp.ndarray): Numerical observation tensor structured as (..., in_features).

        Returns:
            jnp.ndarray: Consistency-verified physical truth interval tensor shaped as (..., in_features, 2).
        """
        # 1. Evaluate coordinate maps to derive raw logical input potentials (z)
        # These potentials establish standard linear feature interactions prior to entering curved logic spaces.
        z_l = self.slope_l[...] * (x - self.offset_l[...])
        z_u = self.slope_u[...] * (x - self.offset_u[...])

        # 2. Project calculated potentials onto the entropic gravitational field activation.
        # When evaluating under 'ramp' geometry, we supply fixed scale factors (slope=1.0, offset=0.5)
        # to the internal activation function because parametric tuning has already been fully
        # absorbed during the potential (z) calculation phase above.
        lower = activations.gravitational_bend_activation(
            z_l, gamma=self.gamma, mode=self.mode, slope=1.0, offset=0.5
        )
        upper = activations.gravitational_bend_activation(
            z_u, gamma=self.gamma, mode=self.mode, slope=1.0, offset=0.5
        )
        
        # 3. Secure axiomatic consistency constraints and pack the resulting tensor slices
        combined = intervals.create_interval(lower, upper)
        return intervals.ensure_interval(combined)
    

class FixedPredicate(nnx.Module):
    """
    Stateless non-trainable identity predicate that preserves input truth intervals.

    This module acts as a rigid, static pass-through transformation layer for pre-computed 
    truth intervals. It is primarily applied in hybrid neuro-symbolic systems, crisp Boolean 
    boundary injections, or deterministic logic anchoring pipelines where structural parameters 
    must remain completely shielded from gradient descent variations.

    Attributes:
        None
    """
    def __init__(self):
        """Initializes the stateless FixedPredicate structural identity layer."""
        pass

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Routes the truth interval tensor forward without altering internal value matrices.

        Args:
            x (jnp.ndarray): Presettled input truth interval tensor structured as (..., 2) 
                representing explicit [Lower, Upper] bounds.

        Returns:
            jnp.ndarray: The identical, unaltered truth interval tensor structured as (..., 2).
        """
        return x  # Identity mapping – [L, U] maps directly to [L, U]