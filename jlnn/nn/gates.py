#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from flax import nnx
from jlnn.core import logic, intervals


class WeightedOr(nnx.Module):
    """
    Trainable weighted OR gate for LNN (Logical Neural Networks).

    This gate implements a weighted version of the Łukasiewicz disjunction. 
    In the LNN architecture, the weights and threshold (beta) 
    are trainable parameters that allow the gate to "learn" logical relationships from the data.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """
        Initializes the parameters of the weighted OR gate.

        Args:
            num_inputs (int): Number of input logical statements (input dimension).
            rngs (nnx.Rngs): Random number generator for initialization (requires Flax NNX).
        """
        # Weights in LNN are typically initialized to 1.0, which corresponds to standard logic
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        # Beta (threshold) determines how true the inputs must be for the gate to turn on
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs a weighted disjunction calculation over the input intervals.

        Args:
            x (jnp.ndarray): Input interval tensor of the form (..., num_inputs, 2). 
                            The last dimension contains [L, U].

        Returns:
            jnp.ndarray: Output interval of the form (..., 2) 
                        representing the truth of the disjunction (OR).
        """
        return logic.weighted_or_lukasiewicz(x, self.weights, self.beta)


class WeightedAnd(nnx.Module):
    """
    Trainable weighted AND gate for LNN.

    It implements weighted conjunction. 
    In LNN, the AND gate is dual to the OR gate according to De Morgan's laws, 
    but here we implement it directly using weighted Łukasiewicz conjunction for 
    higher numerical stability.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """
        Initializes the parameters of the weighted AND gate.

        Args:
            num_inputs (int): Number of input logical statements.
            rngs (nnx.Rngs): Random number generator for initialization.
        """
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates a weighted conjunction over the input intervals.

        Args:
            x (jnp.ndarray): Input interval tensor of the form (..., num_inputs, 2).

        Returns:
            jnp.ndarray: Output interval of the form (..., 2) 
                        representing the truth of the conjunction (AND).
        """
        # Direct call to logical function for AND
        return logic.weighted_and_lukasiewicz(x, self.weights, self.beta)