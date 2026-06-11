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
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return F.weighted_and(x, self.weights[...], self.beta[...])


class WeightedNand(nnx.Module):
    """
    Weighted NAND gate (Negated AND).
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return F.weighted_nand(x, self.weights[...], self.beta[...])


class WeightedNor(nnx.Module):
    """
    Weighted NOR gate (Negated OR).
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return F.weighted_nor(x, self.weights[...], self.beta[...])


class WeightedXor(nnx.Module):
    """
    Trainable n-ary XOR gate implemented via recursive tree reduction.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        if num_inputs < 2:
            raise ValueError("An XOR gate requires at least 2 inputs.")
        self.num_inputs = num_inputs

        if num_inputs == 2:
            self.weights_or = nnx.Param(jnp.ones((2,)))
            self.weights_nand = nnx.Param(jnp.ones((2,)))
            self.weights_and = nnx.Param(jnp.ones((2,)))
            self.beta = nnx.Param(jnp.array(1.0))
            self.left_child = None
            self.right_child = None
        else:
            # Tree reduction setup
            mid = num_inputs // 2
            self.left_child = WeightedXor(num_inputs=mid, rngs=rngs)
            self.right_child = WeightedXor(num_inputs=num_inputs - mid, rngs=rngs)
            self.combiner = WeightedXor(num_inputs=2, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.num_inputs == 2:
            res_or = F.weighted_or(x, self.weights_or[...], self.beta[...])
            res_nand = F.weighted_nand(x, self.weights_nand[...], self.beta[...])
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
    Trainable or self-adaptive implication gate (A -> B).

    Supports standard logical semantics (Łukasiewicz, Reichenbach, Kleene-Dienes, Goguen, Gödel)
    as well as advanced space-curved Physical Fuzzy Logic (PFL) variants.
    """
    def __init__(self, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """
        Initializes the implication gate.

        Args:
            rngs (nnx.Rngs): Flax NNX random number generator collection.
            method (str): Logical method to use ('lukasiewicz', 'kleene_dienes', 'reichenbach', 
                'goguen', 'godel', 'physical_kleene_dienes', 'physical_reichenbach', 'physical_lukasiewicz').
        """
        self.method = method
        
        # Physical methods do not utilize optimization weights as they bend the space dynamically 
        # using entropy. We only instantiate trainable parameters for traditional methods.
        if not self.method.startswith('physical_'):
            self.weights = nnx.Param(jnp.ones((2,)))
            self.beta = nnx.Param(jnp.array(1.0))
        else:
            self.weights = None
            self.beta = None

    def __call__(self, int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the logical implication between two truth intervals.
        """
        if self.weights is not None:
            w = self.weights[...]
            b = self.beta[...]
        else:
            # For physical rules, we feed neutral weights to F.weighted_implication 
            # so that it bypasses the preprocessing step without distorting values or throwing errors.
            w = jnp.ones((2,))
            b = jnp.array(1.0)

        return F.weighted_implication(
            int_a, int_b, w, b, self.method
        )


class WeightedNot(nnx.Module):
    """
    Trainable weighted negation (NOT) gate.

    Allows the model to learn the degree of certainty or strictness 
    associated with logical inversion.
    """
    def __init__(self, rngs: nnx.Rngs):
        self.weight = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return F.weighted_not(x, self.weight[...])