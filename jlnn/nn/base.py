#!/usr/bin/env python3

# Imports
from flax import nnx
import jax.numpy as jnp
from typing import Any


class LogicalElement(nnx.Module):
    """
    The basic abstract class for logical elements within JLNN.
    
    This class serves as the base for all logical gates (AND, OR, Implicature) 
    and predicates. It provides initialization and management of trainable parameters such 
    as weights and thresholds (beta), in accordance with the Logical Neural Networks (LNN) architecture.
    
    Within LNN, we work with interval logic, where each input and output represents 
    a truth value as an interval [Lower Bound, Upper Bound].

    Attributes:
        weights (nnx.Param): Trainable weights for each input. 
            In LNN, they are initialized to 1.0 (neutral influence).
        beta (nnx.Param): Trainable gate bias, 
            determining the steepness of the logic activation.

    Raises:
        NotImplementedError: _description_
    """    
    
    
    def __init__(self, n_inputs: int, rngs: nnx.Rngs):
        """Initializes the parameters of the logical element.

        Args:
            n_inputs (int): Number of input channels (connections) to this element.
            rngs (nnx.Rngs): Collection of random number generators for Flax NNX.
        """        
        # Initialize weights: In LNN we start at 1.0 (neutral weight)
        # We use nnx.Param so that JAX knows what to train
        self.weights = nnx.Param(jnp.ones((n_inputs,)))
        # Threshold (bias) beta
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Abstract method for performing a logical operation.

        Args:
            x (jnp.ndarray): Input interval tensor of the form (..., n_inputs, 2). 
                The last dimension contains [L, U].

        Raises:
            jnp.ndarray: The resulting truth interval [L, U].

        Returns:
            NotImplementedError: This method must be implemented in a specific gate.
        """        
        # This method is implemented by specific gates (AND, OR...)
        raise NotImplementedError

