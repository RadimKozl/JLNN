#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from flax import nnx
from jlnn.nn import functional as F

# =====================================================================
# 1. TRADITIONAL PARAMETRIC LOGIC GATES (Learning Weights and Beta)
# =====================================================================

class WeightedOr(nnx.Module):
    """
    Trainable parametric fuzzy OR gate supporting standard t-conorms.

    In the JLNN framework, this module acts as a stateful neural layer aggregating 
    truth intervals across multiple input streams. It optimizes relative feature 
    importance via weights and shifts the collective activation threshold via a 
    learnable beta parameter.

    Attributes:
        method (str): Target logical framework selector ('lukasiewicz', 'kleene_dienes', 'reichenbach', 'godel', 'product').
        weights (nnx.Param): Trainable input importance weights structured as (num_inputs,).
        beta (nnx.Param): Trainable gate sensitivity threshold scalar parameter (bias).
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """Initializes the stateful WeightedOr gate with corresponding optimization parameters."""
        self.method = method
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Executes the fuzzy OR operation across the designated terminal feature dimension."""
        if self.method == 'lukasiewicz':
            return F.weighted_or(x, self.weights[...], self.beta[...])
        elif self.method in ('godel', 'kleene_dienes'):
            return F.weighted_or_kleene_dienes(x, self.weights[...])
        elif self.method in ('product', 'reichenbach'):
            return F.weighted_or_reichenbach(x, self.weights[...])
        else:
            raise ValueError(f"Parametric OR method '{self.method}' is not supported.")


class WeightedAnd(nnx.Module):
    """
    Trainable parametric fuzzy AND gate supporting standard t-norms.

    Acts as a stateful neural intersection layer evaluating systemic conjunction across 
    multiple truth interval inputs. It minimizes joint evidence through parameterized 
    linear optimizations via weights and beta selection.

    Attributes:
        method (str): Target logical framework selector ('lukasiewicz', 'kleene_dienes', 'reichenbach', 'godel', 'product').
        weights (nnx.Param): Trainable importance weights scaling individual inputs.
        beta (nnx.Param): Trainable gate activation threshold bias parameter.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """Initializes the stateful WeightedAnd gate."""
        self.method = method
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Executes the fuzzy AND operation across the designated terminal feature dimension."""
        if self.method == 'lukasiewicz':
            return F.weighted_and(x, self.weights[...], self.beta[...])
        elif self.method in ('godel', 'kleene_dienes'):
            return F.weighted_and_kleene_dienes(x, self.weights[...])
        elif self.method in ('product', 'reichenbach'):
            return F.weighted_and_reichenbach(x, self.weights[...])
        else:
            raise ValueError(f"Parametric AND method '{self.method}' is not supported.")


class WeightedNand(nnx.Module):
    """
    Trainable parametric fuzzy NAND gate.

    Computes a parameterized multi-variable intersection scaled by dynamic feature importance 
    and sensitivity thresholds, followed by an axiomatic boundary negation to 
    project a final strict truth interval.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        self.method = method
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.method == 'lukasiewicz':
            return F.weighted_nand(x, self.weights[...], self.beta[...])
        elif self.method in ('godel', 'kleene_dienes'):
            return F.weighted_nand_kleene_dienes(x, self.weights[...])
        elif self.method in ('product', 'reichenbach'):
            return F.weighted_nand_reichenbach(x, self.weights[...])
        else:
            raise ValueError(f"Parametric NAND method '{self.method}' is not supported.")


class WeightedNor(nnx.Module):
    """
    Trainable parametric fuzzy NOR gate.

    Computes a parameterized multi-variable union scaled by dynamic feature importance 
    and sensitivity thresholds, followed by an axiomatic boundary negation.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        self.method = method
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.method == 'lukasiewicz':
            return F.weighted_nor(x, self.weights[...], self.beta[...])
        elif self.method in ('godel', 'kleene_dienes'):
            return F.weighted_nor_kleene_dienes(x, self.weights[...])
        elif self.method in ('product', 'reichenbach'):
            return F.weighted_nor_reichenbach(x, self.weights[...])
        else:
            raise ValueError(f"Parametric NOR method '{self.method}' is not supported.")


class WeightedXor(nnx.Module):
    """
    Trainable parametric fuzzy XOR (Exclusive OR) gate.
    """
    def __init__(self, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        self.method = method
        self.weights = nnx.Param(jnp.ones((2,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
        if self.method == 'lukasiewicz':
            return F.weighted_xor_lukasiewicz(int_a, int_b, self.weights[...], self.beta[...])
        elif self.method in ('godel', 'kleene_dienes'):
            return F.weighted_xor_godel(int_a, int_b, self.weights[...])
        elif self.method in ('product', 'reichenbach'):
            return F.weighted_xor_product(int_a, int_b, self.weights[...])
        else:
            raise ValueError(f"Parametric XOR method '{self.method}' is not supported.")


class WeightedNot(nnx.Module):
    """
    Trainable parametric fuzzy NOT inversion gate with adjustable confidence scaling.
    """
    def __init__(self, rngs: nnx.Rngs):
        self.weight = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return F.weighted_not(x, self.weight[...])


# =====================================================================
# 2. PARAMETER-FREE PURE (BULK) REDUCTION GATES
# =====================================================================

class BulkAnd(nnx.Module):
    """
    Non-trainable pure stateless bulk AND reduction gate.

    Directly collapses an input feature dimension using non-parametric, pure analytical 
    fuzzy logic formulas.
    """
    def __init__(self, method: str = 'kleene_dienes'):
        self.method = method

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.method in ('godel', 'kleene_dienes'):
            return F.bulk_and_godel(x)
        elif self.method in ('product', 'reichenbach'):
            return F.bulk_and_product(x)
        elif self.method == 'lukasiewicz':
            inputs = [x[..., i, :] for i in range(x.shape[-2])]
            res = inputs[0]
            for item in inputs[1:]:
                res = F.and_lukasiewicz(res, item)
            return res
        else:
            raise ValueError(f"Bulk AND method '{self.method}' is not recognized.")


class BulkOr(nnx.Module):
    """
    Non-trainable pure stateless bulk OR reduction gate.

    Directly collapses an input feature dimension using non-parametric, pure analytical 
    fuzzy logic formulas.
    """
    def __init__(self, method: str = 'kleene_dienes'):
        self.method = method

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.method in ('godel', 'kleene_dienes'):
            return F.bulk_or_godel(x)
        elif self.method in ('product', 'reichenbach'):
            return F.bulk_or_product(x)
        elif self.method == 'lukasiewicz':
            inputs = [x[..., i, :] for i in range(x.shape[-2])]
            res = inputs[0]
            for item in inputs[1:]:
                res = F.or_lukasiewicz(res, item)
            return res
        else:
            raise ValueError(f"Bulk OR method '{self.method}' is not recognized.")


# =====================================================================
# 3. SPECIAL PHYSICAL GATES (Physical Fuzzy Logic - PFL)
# =====================================================================

class PhysicalOr(nnx.Module):
    """
    Space-curved entropic physical OR gate with localized field configurations.
    """
    def __init__(self, method: str = 'physical_kleene_dienes', gamma: float = 0.2, 
                 mode: str = 'sigmoid', slope: float = 1.0, offset: float = 0.5):
        self.method = method
        self.gamma = gamma
        self.mode = mode
        self.slope = slope
        self.offset = offset

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        inputs = [x[..., i, :] for i in range(x.shape[-2])]
        res = inputs[0]
        
        if self.method in ('physical_godel', 'physical_kleene_dienes'):
            for item in inputs[1:]:
                res = F.or_physical_kleene_dienes(res, item)
            return res
        elif self.method in ('physical_product', 'physical_reichenbach'):
            for item in inputs[1:]:
                res = F.or_physical_reichenbach(res, item)
            return res
        elif self.method == 'physical_lukasiewicz':
            for item in inputs[1:]:
                res = F.or_physical_lukasiewicz(res, item)
            return res
        else:
            raise ValueError(f"Physical OR method '{self.method}' is not supported.")


class PhysicalAnd(nnx.Module):
    """
    Space-curved entropic physical AND gate with localized field configurations.
    """
    def __init__(self, method: str = 'physical_kleene_dienes', gamma: float = 0.2, 
                 mode: str = 'sigmoid', slope: float = 1.0, offset: float = 0.5):
        self.method = method
        self.gamma = gamma
        self.mode = mode
        self.slope = slope
        self.offset = offset

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        inputs = [x[..., i, :] for i in range(x.shape[-2])]
        res = inputs[0]
        
        if self.method in ('physical_godel', 'physical_kleene_dienes'):
            for item in inputs[1:]:
                res = F.and_physical_kleene_dienes(res, item)
            return res
        elif self.method in ('physical_product', 'physical_reichenbach'):
            for item in inputs[1:]:
                res = F.and_physical_reichenbach(res, item)
            return res
        elif self.method == 'physical_lukasiewicz':
            for item in inputs[1:]:
                res = F.and_physical_lukasiewicz(res, item)
            return res
        else:
            raise ValueError(f"Physical AND method '{self.method}' is not supported.")


class PhysicalImplication(nnx.Module):
    """
    Parameter-free space-curved rule gateway (A -> B).
    """
    def __init__(self, method: str = 'physical_kleene_dienes', gamma: float = 0.2, 
                 mode: str = 'sigmoid', slope: float = 1.0, offset: float = 0.5):
        self.method = method
        self.gamma = gamma
        self.mode = mode
        self.slope = slope
        self.offset = offset

    def __call__(self, int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
        """Computes pure physical implication directly using the underlying API contract."""
        return F.implication(int_a, int_b, method=self.method)


class PhysicalNot(nnx.Module):
    """
    Parameter-free physical inversion (NOT) gate.
    """
    def __init__(self):
        pass

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return F.logical_not(x)


class PhysicalNand(nnx.Module):
    """
    Parameter-free physical NAND gate (Alternative Denial).
    """
    def __init__(self, method: str = 'physical_kleene_dienes', gamma: float = 0.2, 
                 mode: str = 'sigmoid', slope: float = 1.0, offset: float = 0.5):
        self.and_gate = PhysicalAnd(method=method, gamma=gamma, mode=mode, slope=slope, offset=offset)
        self.not_gate = PhysicalNot()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        and_res = self.and_gate(x)
        return self.not_gate(and_res)


class PhysicalNor(nnx.Module):
    """
    Parameter-free physical NOR gate (Joint Denial).
    """
    def __init__(self, method: str = 'physical_kleene_dienes', gamma: float = 0.2, 
                 mode: str = 'sigmoid', slope: float = 1.0, offset: float = 0.5):
        self.or_gate = PhysicalOr(method=method, gamma=gamma, mode=mode, slope=slope, offset=offset)
        self.not_gate = PhysicalNot()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        or_res = self.or_gate(x)
        return self.not_gate(or_res)