#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from flax import nnx
from jlnn.nn import functional as F

class WeightedOr(nnx.Module):
    """
    Trainable or self-adaptive OR gate supporting multiple logical semantics.

    In the JLNN framework, this gate aggregates truth intervals from multiple inputs.
    For standard semantics, it learns input importance via weights and sensitivity via beta.
    For physical semantics (PFL), it operates in a parameter-free manner guided by entropy.

    Attributes:
        weights (Optional[nnx.Param]): Importance weights for each input signal.
        beta (Optional[nnx.Param]): Sensitivity threshold (bias) of the disjunction.
        method (str): Logical semantics ('lukasiewicz', 'godel', 'product', 'drastic', 'physical_kleene_dienes').
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        self.method = method
        
        # Physical methods (PFL) are inherently parameter-free.
        if not self.method.startswith('physical_'):
            self.weights = nnx.Param(jnp.ones((num_inputs,)))
            self.beta = nnx.Param(jnp.array(1.0))
        else:
            self.weights = None
            self.beta = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the OR operation across the input dimension.
        """
        if self.method == 'lukasiewicz':
            return F.weighted_or(x, self.weights[...], self.beta[...])
        elif self.method == 'godel':
            return F.weighted_or_godel(x, self.weights[...])
        elif self.method == 'product':
            return F.weighted_or_product(x, self.weights[...])
        elif self.method == 'drastic':
            # Drastic logic reduction uses the bulk/pure stateless implementation
            return F.bulk_or_drastic(x)
        elif self.method == 'physical_kleene_dienes':
            # Physical OR operates binary reductions or can be mapped natively.
            # For multi-input consistency, we reduce using the core physical binary operator.
            # JAX reduction across the input features axis (-2)
            def reduce_fn(carry, current):
                return F.or_physical_kleene_dienes(carry, current)
            
            # Unstack along the input features axis to iterate
            inputs = [x[..., i, :] for i in range(x.shape[-2])]
            res = inputs[0]
            for item in inputs[1:]:
                res = reduce_fn(res, item)
            return res
        else:
            raise ValueError(f"OR method '{self.method}' is not supported.")


class WeightedAnd(nnx.Module):
    """
    Trainable or self-adaptive AND gate supporting multiple logical semantics.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        self.method = method
        
        if not self.method.startswith('physical_'):
            self.weights = nnx.Param(jnp.ones((num_inputs,)))
            self.beta = nnx.Param(jnp.array(1.0))
        else:
            self.weights = None
            self.beta = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the AND operation across the input dimension.
        """
        if self.method == 'lukasiewicz':
            return F.weighted_and(x, self.weights[...], self.beta[...])
        elif self.method == 'godel':
            return F.weighted_and_godel(x, self.weights[...])
        elif self.method == 'product':
            return F.weighted_and_product(x, self.weights[...])
        elif self.method == 'drastic':
            return F.bulk_and_drastic(x)
        elif self.method == 'physical_kleene_dienes':
            # Sequential physical reduction across the feature dimensions
            inputs = [x[..., i, :] for i in range(x.shape[-2])]
            res = inputs[0]
            for item in inputs[1:]:
                res = F.and_physical_kleene_dienes(res, item)
            return res
        else:
            raise ValueError(f"AND method '{self.method}' is not supported.")


class BulkAnd(nnx.Module):
    """
    Non-trainable pure stateless bulk AND reduction gate.
    Directly collapses a feature dimension using structural structural logical semantics.
    """
    def __init__(self, method: str = 'godel'):
        self.method = method

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.method == 'godel':
            return F.bulk_and_godel(x)
        elif self.method == 'product':
            return F.bulk_and_product(x)
        elif self.method == 'drastic':
            return F.bulk_and_drastic(x)
        else:
            raise ValueError(f"Bulk AND method '{self.method}' is not recognized or requires weights.")


class BulkOr(nnx.Module):
    """
    Non-trainable pure stateless bulk OR reduction gate.
    Directly collapses a feature dimension using structural logical semantics.
    """
    def __init__(self, method: str = 'godel'):
        self.method = method

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.method == 'godel':
            return F.bulk_or_godel(x)
        elif self.method == 'product':
            return F.bulk_or_product(x)
        elif self.method == 'drastic':
            return F.bulk_or_drastic(x)
        else:
            raise ValueError(f"Bulk OR method '{self.method}' is not recognized or requires weights.")


class WeightedNand(nnx.Module):
    """
    Weighted NAND gate (Negated AND) utilizing Łukasiewicz core.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return F.weighted_nand(x, self.weights[...], self.beta[...])


class WeightedNor(nnx.Module):
    """
    Weighted NOR gate (Negated OR) utilizing Łukasiewicz core.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return F.weighted_nor(x, self.weights[...], self.beta[...])


class WeightedXor(nnx.Module):
    """
    Trainable or structural n-ary XOR gate implemented via recursive tree reduction.
    Supports Łukasiewicz, Gödel, and Product logic XOR variants.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        if num_inputs < 2:
            raise ValueError("An XOR gate requires at least 2 inputs.")
        self.num_inputs = num_inputs
        self.method = method

        if num_inputs == 2:
            if self.method == 'lukasiewicz':
                self.weights_or = nnx.Param(jnp.ones((2,)))
                self.weights_nand = nnx.Param(jnp.ones((2,)))
                self.weights_and = nnx.Param(jnp.ones((2,)))
                self.beta = nnx.Param(jnp.array(1.0))
            else:
                # Godel and Product XOR use analytical pure functions from functional.py
                self.weights_or = None
        else:
            # Recursive tree division
            mid = num_inputs // 2
            self.left_child = WeightedXor(num_inputs=mid, rngs=rngs, method=method)
            self.right_child = WeightedXor(num_inputs=num_inputs - mid, rngs=rngs, method=method)
            self.combiner = WeightedXor(num_inputs=2, rngs=rngs, method=method)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.num_inputs == 2:
            # Access the individual binary interval components
            int_a = x[..., 0, :]
            int_b = x[..., 1, :]
            
            if self.method == 'lukasiewicz':
                res_or = F.weighted_or(x, self.weights_or[...], self.beta[...])
                res_nand = F.weighted_nand(x, self.weights_nand[...], self.beta[...])
                combined = jnp.stack([res_or, res_nand], axis=-2)
                return F.weighted_and(combined, self.weights_and[...], self.beta[...])
            elif self.method == 'godel':
                return F.xor_godel(int_a, int_b)
            elif self.method == 'product':
                return F.xor_product(int_a, int_b)
            else:
                raise ValueError(f"XOR method '{self.method}' is not supported.")
        else:
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
        self.method = method
        
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
            # Neutral mask bypass to keep function signature uniform
            w = jnp.ones((2,))
            b = jnp.array(1.0)

        return F.weighted_implication(
            int_a, int_b, w, b, self.method
        )


class WeightedNot(nnx.Module):
    """
    Trainable weighted negation (NOT) gate with adjustable structural confidence scaling.
    """
    def __init__(self, rngs: nnx.Rngs):
        self.weight = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return F.weighted_not(x, self.weight[...])