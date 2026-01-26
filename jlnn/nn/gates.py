#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from flax import nnx
from jlnn.nn import functional as F

class WeightedOr(nnx.Module):
    """
    Trainable weighted OR gate implemented using Łukasiewicz t-conorm.
    
    In the JLNN framework, this gate aggregates the truth intervals of the inputs 
    and allows the model to learn the importance (weight) of each argument 
    and the sensitivity threshold (beta) for the disjunction. 
    The computation is delegated to a stateless function in the functional module.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """
        Initializes the OR gate parameters.

        Args:
            num_inputs (int): Number of input logic signals.
            rngs (nnx.Rngs): Random number generator for initializing NNX parameters.
        """
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs a forward calculation of the weighted disjunction.

        Args:
            x (jnp.ndarray): Input interval tensor of the form (..., num_inputs, 2).

        Returns:
            jnp.ndarray: The resulting aggregated truth interval [L, U].
        """
        return F.weighted_or(x, self.weights.value, self.beta.value)


class WeightedAnd(nnx.Module):
    """
    Trainable weighted AND gate implemented using Łukasiewicz's t-norm.
    
    This gate implements logical conjunction over intervals. 
    The weight parameters allow the network to selectively suppress unimportant inputs, 
    while the beta parameter controls the stringency of the logical product.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """
        Initializes the AND gate parameters.

        Args:
            num_inputs (int): Number of input logic signals.
            rngs (nnx.Rngs): Generator for NNX state.
        """
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs a forward calculation of the weighted conjunction.

        Args:
            x (jnp.ndarray): Input interval tensor of the form (..., num_inputs, 2).

        Returns:
            jnp.ndarray: The resulting aggregated truth interval [L, U].
        """
        return F.weighted_and(x, self.weights.value, self.beta.value)


class WeightedNand(nnx.Module):
    """
    Weighted NAND gate (Negated AND).
    
    Implements negation of conjunction. In JLNN, 
    this operation is key for enforcing constraints, 
    where we want to prevent two contradictory statements 
    from being true at the same time.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes a negated weighted conjunction using a functional interface.
        """
        return F.weighted_nand(x, self.weights.value, self.beta.value)


class WeightedNor(nnx.Module):
    """
    Weighted NOR gate (Negated OR).
    
    This operator returns a high truth value only if 
    all weighted inputs have a low truth value. 
    It is often used to detect the absence of specific features in the data.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the negated weighted disjunction.
        """
        return F.weighted_nor(x, self.weights.value, self.beta.value)


class WeightedXor(nnx.Module):
    """
    Trainable n-ary XOR gate (Exclusive OR) using tree reduction.
    
    XOR is implemented in interval logic as a recursive binary tree. 
    For n=2 it uses the composition (A OR B) AND (A NAND B). 
    For n > 2 the inputs are divided and the results are combined using binary XOR operations. 
    This structure allows learning complex parity functions with trainable weights at each node of the tree.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """
        It builds a hierarchical structure of an XOR gate.

        Args:
            num_inputs (int): Total number of inputs.
            rngs (nnx.Rngs): Generator for initializing all internal gates.
        """
        if num_inputs < 2:
            raise ValueError("An XOR gate requires at least 2 inputs.")
        self.num_inputs = num_inputs

        if num_inputs == 2:
            # Base case: parameters for internal logical composition XOR
            self.weights_or = nnx.Param(jnp.ones((2,)))
            self.weights_nand = nnx.Param(jnp.ones((2,)))
            self.weights_and = nnx.Param(jnp.ones((2,)))
            self.beta = nnx.Param(jnp.array(1.0))
            self.left_child = None
            self.right_child = None
        else:
            # Recursive case: splitting inputs and creating subtrees
            mid = num_inputs // 2
            self.left_child = WeightedXor(num_inputs=mid, rngs=rngs)
            self.right_child = WeightedXor(num_inputs=num_inputs - mid, rngs=rngs)
            self.combiner = WeightedXor(num_inputs=2, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs a recursive parity calculation (XOR) over the tree structure.
        """
        if self.num_inputs == 2:
            # Atomic binary XOR operation using functional.py
            res_or = F.weighted_or(x, self.weights_or.value, self.beta.value)
            res_nand = F.weighted_nand(x, self.weights_nand.value, self.beta.value)
            combined = jnp.stack([res_or, res_nand], axis=-2)
            return F.weighted_and(combined, self.weights_and.value, self.beta.value)
        else:
            # Traversal of the reduction tree
            mid = self.num_inputs // 2
            res_left = self.left_child(x[..., :mid, :])
            res_right = self.right_child(x[..., mid:, :])
            combined_results = jnp.stack([res_left, res_right], axis=-2)
            return self.combiner(combined_results)


class WeightedImplication(nnx.Module):
    """
    Trainable gate for logical implication (A -> B).
    
    Allows modeling of causal relationships between statements. 
    Supports multiple semantics (Łukasiewicz, Reichenbach, Kleene-Dienes) via the 'method' parameter. 
    Weights allow setting different importance of antecedent and consequent.
    """
    def __init__(self, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """
        Args:
            rngs (nnx.Rngs): Generator for parameters.
            method (str): Selected logical semantics for implication.
        """
        self.method = method
        self.weights = nnx.Param(jnp.ones((2,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the truth interval of the implication.
        """
        return F.weighted_implication(
            int_a, int_b, self.weights.value, self.beta.value, self.method
        )
        
        
class WeightedNot(nnx.Module):
    """
    Trainable weighted negation (NOT) gate.
    
    The weight parameter allows the network to learn how strongly to invert a given input. 
    This is useful in situations where negating a statement is only partially relevant 
    or under certain conditions.
    """
    def __init__(self, rngs: nnx.Rngs):
        self.weight = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies a weighted negation to the input interval.
        """
        return F.weighted_not(x, self.weight.value)