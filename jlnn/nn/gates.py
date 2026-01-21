#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from flax import nnx
from jlnn.core import logic, intervals

class WeightedOr(nnx.Module):
    """
    Trainable weighted OR gate for LNN (Logical Neural Networks).

    This gate implements a weighted version of the Łukasiewicz disjunction (t-conorms). 
    Within the JLNN architecture, the weights and threshold (beta) are defined as optimizable parameters (nnx.Param), 
    allowing the logic gate to learn the importance of individual inputs directly from the data.

    The gate works with interval logic, 
    where the operation is performed independently of the lower (L) and upper (U) truth limits.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """
        Initializes the parameters of the weighted OR gate.

        Args:
            num_inputs (int): Number of input logical arguments (dimension of the input layer).
            rngs (nnx.Rngs): Random number generator required to initialize Flax NNX state.
        """
        # The weights are initialized to 1.0, which corresponds to the standard logical sum.
        # In LNN, it is recommended to keep weights >= 1 to maintain logical interpretability.
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        
        # Beta (threshold/bias) determines the sensitivity of the gate.
        # A value of 1.0 indicates standard Łukasiewicz semantics.
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs a forward pass of the weighted disjunction.

        The calculation is performed according to the formula: f(x) = min(1, sum(w_i * x_i) / beta).

        Args:
            x (jnp.ndarray): Input interval tensor of the form (..., num_inputs, 2). 
                            The last dimension contains the pair [Lower Bound, Upper Bound].

        Returns:
            jnp.ndarray: Output truth interval of the form (..., 2).
        """
        # Delegating computation to a low-level implementation in core.logic
        return logic.weighted_or_lukasiewicz(x, self.weights, self.beta)


class WeightedAnd(nnx.Module):
    """
    Trainable weighted AND gate for LNN (Logical Neural Networks).
    
    This gate implements a weighted version of the Łukasiewicz conjunction (t-norm). 
    Within the JLNN architecture, it is used to aggregate conditions that must be met simultaneously. 
    By using nnx.Param, the weights and threshold (beta) are optimizable, 
    allowing the network to learn the relevance of individual inputs to the resulting conjunction.
    
    The gate directly operates on the truth intervals [L, U], thereby preserving information 
    about uncertainty (epistemic uncertainty) across the computational graph.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """
        Initializes the parameters of the weighted AND gate.

        Args:
            num_inputs (int): Number of input logical arguments (flags).
            rngs (nnx.Rngs): Random number generator for initializing the Flax NNX state.
        """
        
        # Weights are initialized to 1.0. A higher weight for a particular input
        # means that its (false) truth has a stronger influence on the result of the conjunction.
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        
        # Beta (threshold) determines the stringency of the gate. Higher beta dampens the influence of
        # negative evidence on the resulting truth.
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs a forward calculation of the weighted conjunction.

        The calculation is carried out according to the formula: 1 - min(1, sum(w_i * (1 - x_i)) / beta).

        Args:
            x (jnp.ndarray): Input interval tensor of the form (..., num_inputs, 2). 
                            The last dimension contains the pair [Lower Bound, Upper Bound].

        Returns:
            jnp.ndarray: Output truth interval of the form (..., 2).
        """
        # Calling a low-level function from jlnn.core.logic for efficient computation
        return logic.weighted_and_lukasiewicz(x, self.weights, self.beta)


class WeightedImplication(nnx.Module):
    """
    Trainable gate for logical implication (A -> B).
    
    This gate implements the relationship between antecedent (premise A) 
    and consequent (consequence B). Within JLNN, it supports multiple semantics, 
    allowing the user to choose between optimistic (Łukasiewicz), pessimistic (Kleene-Dienes), 
    or compromise (Reichenbach) approaches.
    
    The weights on the implication allow the model to learn the relevance 
    of individual parts of the rule, while the beta parameter controls 
    the strictness of the activation of the entire rule.
    """
    def __init__(self, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """
        Initializes the parameters of the weighted implication.

        Args:
            rngs (nnx.Rngs): Random number generator for Flax NNX.
            method (str): Selected logical method. 
                        Supported values: 'lukasiewicz', 'kleene_dienes', 'reichenbach'.
        """
        self.method = method
        
        # An implication has 2 inputs: an antecedent (A) and a consequent (B).
        # The weights are initialized to 1.0, which corresponds to the standard logical strength.
        self.weights = nnx.Param(jnp.ones((2,)))
        # Beta determines the sensitivity threshold for activating the implication.
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the truth interval of an implication between two statements.

        Args:
            int_a (jnp.ndarray): Tensor for the antecedent (A) of the form (..., 2). 
            int_b (jnp.ndarray): Tensor for the consequent (B) of the form (..., 2).

        Returns:
            jnp.ndarray: The resulting truth interval [L, U] of the form (..., 2).
        
        Raises:
            ValueError: If an unsupported calculation method is set.
        """
        if self.method == 'lukasiewicz':
            # Łukasiewicz uses internal weighted OR (¬A ∨ B).
            return logic.implies_lukasiewicz(int_a, int_b, self.weights, self.beta)
        
        # For the Kleene-Dienes and Reichenbach methods, we apply weights as preprocessing.
        # This scales the importance of the input before applying the logical function itself.
        weighted_a = intervals.create_interval(
            jnp.minimum(1.0, intervals.get_lower(int_a) * self.weights[0]),
            jnp.minimum(1.0, intervals.get_upper(int_a) * self.weights[0])
        )
        weighted_b = intervals.create_interval(
            jnp.minimum(1.0, intervals.get_lower(int_b) * self.weights[1]),
            jnp.minimum(1.0, intervals.get_upper(int_b) * self.weights[1])
        )

        if self.method == 'kleene_dienes':
            # max(1 - A, B)
            return logic.implies_kleene_dienes(weighted_a, weighted_b)
        elif self.method == 'reichenbach':
            # 1 - A + (A * B)
            return logic.implies_reichenbach(weighted_a, weighted_b)
        else:
            raise ValueError(f"Unsupported implication method: {self.method}")