#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from flax import nnx
from jlnn.nn import functional as F

class WeightedOr(nnx.Module):
    """
    Trainable weighted OR gate implemented using Łukasiewicz t-conorm.

    In the JLNN framework, this gate aggregates truth intervals from multiple inputs.
    It learns the relative importance of each input through weights and adjusts
    the activation threshold via the beta parameter.

    Attributes:
        weights (nnx.Param): Importance weights for each input signal.
        beta (nnx.Param): Sensitivity threshold (bias) of the disjunction.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the weighted OR operation.

        Args:
            x (jnp.ndarray): Input interval tensor of shape (..., num_inputs, 2).

        Returns:
            jnp.ndarray: The resulting truth interval [L, U].
        """
        # Note: Using [...] for parameter access ensures compatibility with NNX 
        # and avoids DeprecationWarnings associated with the .value property.
        return F.weighted_or(x, self.weights[...], self.beta[...])


class WeightedAnd(nnx.Module):
    """
    Trainable weighted AND gate implemented using Łukasiewicz t-norm.

    This gate performs a fuzzy conjunction where weights determine how much
    each input contributes to the "negative evidence" against the truth.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Computes the weighted Łukasiewicz conjunction."""
        return F.weighted_and(x, self.weights[...], self.beta[...])


class WeightedNand(nnx.Module):
    """
    Weighted NAND gate (Negated AND).

    Useful for enforcing constraints where two contradictory statements 
    should not be simultaneously true.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return F.weighted_nand(x, self.weights[...], self.beta[...])


class WeightedNor(nnx.Module):
    """
    Weighted NOR gate (Negated OR).

    Evaluates to high truth only if all weighted inputs are close to falsehood.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return F.weighted_nor(x, self.weights[...], self.beta[...])


class WeightedXor(nnx.Module):
    """
    Trainable n-ary XOR gate implemented via recursive tree reduction.

    XOR in interval logic is non-trivial. For n=2, it uses the logical 
    composition: (A OR B) AND (A NAND B). For n > 2, it recursively builds 
    a binary tree of XOR operations.

    This hierarchical structure allows the network to learn complex parity-like 
    functions with independent weights at each node.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        if num_inputs < 2:
            raise ValueError("An XOR gate requires at least 2 inputs.")
        self.num_inputs = num_inputs

        if num_inputs == 2:
            # Base case: Parameters for the internal composition
            self.weights_or = nnx.Param(jnp.ones((2,)))
            self.weights_nand = nnx.Param(jnp.ones((2,)))
            self.weights_and = nnx.Param(jnp.ones((2,)))
            self.beta = nnx.Param(jnp.array(1.0))
            self.left_child = None
            self.right_child = None
        else:
            # Recursive case: Build a balanced tree
            mid = num_inputs // 2
            self.left_child = WeightedXor(num_inputs=mid, rngs=rngs)
            self.right_child = WeightedXor(num_inputs=num_inputs - mid, rngs=rngs)
            self.combiner = WeightedXor(num_inputs=2, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.num_inputs == 2:
            # Atomic binary XOR logic
            res_or = F.weighted_or(x, self.weights_or[...], self.beta[...])
            res_nand = F.weighted_nand(x, self.weights_nand[...], self.beta[...])
            
            # Combine results into a new interval set for the final AND gate
            combined = jnp.stack([res_or, res_nand], axis=-2)
            return F.weighted_and(combined, self.weights_and[...], self.beta[...])
        else:
            # Tree traversal: pass data through sub-modules
            mid = self.num_inputs // 2
            res_left = self.left_child(x[..., :mid, :])
            res_right = self.right_child(x[..., mid:, :])
            combined_results = jnp.stack([res_left, res_right], axis=-2)
            return self.combiner(combined_results)


class WeightedImplication(nnx.Module):
    """
    Trainable implication gate (A -> B).

    Supports multiple semantics (Łukasiewicz, Reichenbach, Kleene-Dienes).
    Ideal for modeling expert-driven rules within the neural architecture.
    """
    def __init__(self, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        self.method = method
        self.weights = nnx.Param(jnp.ones((2,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
        return F.weighted_implication(
            int_a, int_b, self.weights[...], self.beta[...], self.method
        )
        
        
class WeightedNot(nnx.Module):
    """
    Trainable weighted negation (NOT) gate.

    Allows the model to learn the degree of inversion for a specific statement.
    """
    def __init__(self, rngs: nnx.Rngs):
        self.weight = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return F.weighted_not(x, self.weight[...])