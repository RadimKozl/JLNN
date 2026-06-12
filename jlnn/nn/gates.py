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
        method (str): Target logical framework selector ('lukasiewicz', 'godel', 'product').
        weights (nnx.Param): Trainable input importance weights structured as (num_inputs,)
        beta (nnx.Param): Trainable gate sensitivity threshold scalar parameter (bias).
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """
        Initializes the stateful WeightedOr gate with corresponding optimization parameters.

        Args:
            num_inputs (int): Dimensionality of the incoming input feature sub-space.
            rngs (nnx.Rngs): Flax NNX random number generator collection for parameter initialization.
            method (str, optional): Target fuzzy logic framework. Defaults to 'lukasiewicz'.
        """
        self.method = method
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the fuzzy OR operation across the designated terminal feature dimension.

        Args:
            x (jnp.ndarray): Multi-variable truth interval tensor structured as 
                (..., num_inputs, 2), where the last dimension holds [Lower, Upper] bounds.

        Returns:
            jnp.ndarray: Bounded and consistency-verified truth interval structured as (..., 2).

        Raises:
            ValueError: If an unmapped or invalid logic method string is provided.
        """
        if self.method == 'lukasiewicz':
            return F.weighted_or(x, self.weights[...], self.beta[...])
        elif self.method == 'godel':
            return F.weighted_or_godel(x, self.weights[...])
        elif self.method == 'product':
            return F.weighted_or_product(x, self.weights[...])
        else:
            raise ValueError(f"Parametric OR method '{self.method}' is not supported.")


class WeightedAnd(nnx.Module):
    """
    Trainable parametric fuzzy AND gate supporting standard t-norms.

    Acts as a stateful neural intersection layer evaluating systemic conjunction across 
    multiple truth interval inputs. It minimizes joint evidence through parameterized 
    linear optimizations via weights and beta selection.

    Attributes:
        method (str): Target logical framework selector ('lukasiewicz', 'godel', 'product').
        weights (nnx.Param): Trainable importance weights scaling individual inputs.
        beta (nnx.Param): Trainable gate activation threshold bias parameter.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """
        Initializes the stateful WeightedAnd gate.

        Args:
            num_inputs (int): Dimensionality of the incoming input feature sub-space.
            rngs (nnx.Rngs): Flax NNX random number generator collection.
            method (str, optional): Target fuzzy logic framework. Defaults to 'lukasiewicz'.
        """
        self.method = method
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the fuzzy AND operation across the designated terminal feature dimension.

        Args:
            x (jnp.ndarray): Multi-variable truth interval tensor structured as 
                (..., num_inputs, 2), where the last dimension holds [Lower, Upper] bounds.

        Returns:
            jnp.ndarray: Bounded and consistency-verified truth interval structured as (..., 2).

        Raises:
            ValueError: If an unmapped or invalid logic method string is provided.
        """
        if self.method == 'lukasiewicz':
            return F.weighted_and(x, self.weights[...], self.beta[...])
        elif self.method == 'godel':
            return F.weighted_and_godel(x, self.weights[...])
        elif self.method == 'product':
            return F.weighted_and_product(x, self.weights[...])
        else:
            raise ValueError(f"Parametric AND method '{self.method}' is not supported.")


# =====================================================================
# 2. PARAMETER-FREE PURE (BULK) REDUCTION GATES
# =====================================================================

class BulkAnd(nnx.Module):
    """
    Non-trainable pure stateless bulk AND reduction gate.

    Directly collapses an input feature dimension using non-parametric, pure analytical 
    fuzzy logic formulas. This gate is ideal for rigid axiomatic consensus aggregation 
    where parameter tracking is computationally redundant.

    Attributes:
        method (str): Pure semantic framework selector ('godel', 'product', 'drastic').
    """
    def __init__(self, method: str = 'godel'):
        """
        Initializes the stateless BulkAnd gate.

        Args:
            method (str, optional): Pure t-norm framework. Defaults to 'godel'.
        """
        self.method = method

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Collapses the penultimate axis of the input tensor via stateless conjunction.

        Args:
            x (jnp.ndarray): Multi-variable truth interval tensor structured as (..., num_inputs, 2).

        Returns:
            jnp.ndarray: Collapsed consensus truth interval structured as (..., 2).

        Raises:
            ValueError: If the method is unrecognized.
        """
        if self.method == 'godel':
            return F.bulk_and_godel(x)
        elif self.method == 'product':
            return F.bulk_and_product(x)
        elif self.method == 'drastic':
            return F.bulk_and_drastic(x)
        else:
            raise ValueError(f"Bulk AND method '{self.method}' is not recognized.")


class BulkOr(nnx.Module):
    """
    Non-trainable pure stateless bulk OR reduction gate.

    Directly collapses an input feature dimension using non-parametric, pure analytical 
    fuzzy logic formulas. This gate aggregates maximum supportive evidence without tracking 
    backpropagation parameter states.

    Attributes:
        method (str): Pure semantic framework selector ('godel', 'product', 'drastic').
    """
    def __init__(self, method: str = 'godel'):
        """
        Initializes the stateless BulkOr gate.

        Args:
            method (str, optional): Pure t-conorm framework. Defaults to 'godel'.
        """
        self.method = method

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Collapses the penultimate axis of the input tensor via stateless disjunction.

        Args:
            x (jnp.ndarray): Multi-variable truth interval tensor structured as (..., num_inputs, 2).

        Returns:
            jnp.ndarray: Collapsed alternative truth interval structured as (..., 2).

        Raises:
            ValueError: If the method is unrecognized.
        """
        if self.method == 'godel':
            return F.bulk_or_godel(x)
        elif self.method == 'product':
            return F.bulk_or_product(x)
        elif self.method == 'drastic':
            return F.bulk_or_drastic(x)
        else:
            raise ValueError(f"Bulk OR method '{self.method}' is not recognized.")


# =====================================================================
# 3. SPECIAL PHYSICAL GATES (Physical Fuzzy Logic - PFL)
# =====================================================================

class PhysicalOr(nnx.Module):
    """
    Space-curved entropic physical OR gate with localized field configurations.

    Operates entirely without internal trainable neural weights or biases, mapping 
    spatial distortion directly through signal interaction governed by entropic space constants.

    Attributes:
        method (str): Target physical logical framework. Supported: 'physical_kleene_dienes'.
        gamma (float): Bending strength coefficient, bounded within the interval [0, 1].
        mode (str): Base compression strategy ('sigmoid' or 'ramp').
        slope (float): Stringency parameter utilized exclusively when mode='ramp'.
        offset (float): Midpoint shift parameter utilized exclusively when mode='ramp'.
    """
    def __init__(self, method: str = 'physical_kleene_dienes', gamma: float = 0.2, 
                 mode: str = 'sigmoid', slope: float = 1.0, offset: float = 0.5):
        """Initializes the parameter-free PhysicalOr gate with entropic parameters."""
        self.method = method
        self.gamma = gamma
        self.mode = mode
        self.slope = slope
        self.offset = offset

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Executes sequential physical disjunction across the input features axis."""
        inputs = [x[..., i, :] for i in range(x.shape[-2])]
        res = inputs[0]
        
        if self.method == 'physical_kleene_dienes':
            for item in inputs[1:]:
                res = F.or_physical_kleene_dienes(res, item)
            return res
        else:
            raise ValueError(f"Physical OR method '{self.method}' is not supported.")


class PhysicalAnd(nnx.Module):
    """
    Space-curved entropic physical AND gate with localized field configurations.

    Attributes:
        method (str): Target physical logical framework. Supported: 'physical_kleene_dienes'.
        gamma (float): Bending strength coefficient, bounded within the interval [0, 1].
        mode (str): Base compression strategy ('sigmoid' or 'ramp').
        slope (float): Stringency parameter utilized exclusively when mode='ramp'.
        offset (float): Midpoint shift parameter utilized exclusively when mode='ramp'.
    """
    def __init__(self, method: str = 'physical_kleene_dienes', gamma: float = 0.2, 
                 mode: str = 'sigmoid', slope: float = 1.0, offset: float = 0.5):
        """Initializes the parameter-free PhysicalAnd gate with entropic parameters."""
        self.method = method
        self.gamma = gamma
        self.mode = mode
        self.slope = slope
        self.offset = offset

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Executes sequential physical conjunction across the input features axis."""
        inputs = [x[..., i, :] for i in range(x.shape[-2])]
        res = inputs[0]
        
        if self.method == 'physical_kleene_dienes':
            for item in inputs[1:]:
                res = F.and_physical_kleene_dienes(res, item)
            return res
        else:
            raise ValueError(f"Physical AND method '{self.method}' is not supported.")


class PhysicalImplication(nnx.Module):
    """
    Parameter-free space-curved rule gateway (A -> B).

    Evaluates rule integrity by bending truth values around entropic fields 
    configured via continuous gravitational configuration parameters.

    Attributes:
        method (str): Target PFL implication ('physical_kleene_dienes', 'physical_reichenbach', 'physical_lukasiewicz').
        gamma (float): Bending strength coefficient, bounded within the interval [0, 1].
        mode (str): Base compression strategy ('sigmoid' or 'ramp').
        slope (float): Stringency parameter utilized exclusively when mode='ramp'.
        offset (float): Midpoint shift parameter utilized exclusively when mode='ramp'.
    """
    def __init__(self, method: str = 'physical_kleene_dienes', gamma: float = 0.2, 
                 mode: str = 'sigmoid', slope: float = 1.0, offset: float = 0.5):
        """Initializes the parameter-free PhysicalImplication gate layer."""
        self.method = method
        self.gamma = gamma
        self.mode = mode
        self.slope = slope
        self.offset = offset

    def __call__(self, int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
        """Computes the physical log-implication bypassing traditional parametric constraints."""
        w_dummy = jnp.ones((2,))
        b_dummy = jnp.array(1.0)
        return F.weighted_implication(int_a, int_b, w_dummy, b_dummy, self.method)


class PhysicalNot(nnx.Module):
    """
    Parameter-free physical inversion (NOT) gate.

    Operates purely through standard logical negation under space-warped interval boundaries.
    """
    def __init__(self):
        """Initializes the parameter-free PhysicalNot inversion gate."""
        pass

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Executes physical negation using a default standard scale factor of 1.0."""
        w_dummy = jnp.array(1.0)
        return F.weighted_not(x, w_dummy)


class PhysicalNand(nnx.Module):
    """
    Parameter-free physical NAND gate (Alternative Denial).

    Evaluates sequential physical conjunction across the inputs and subsequently 
    applies physical axiomatic boundary negation.
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

    Evaluates sequential physical disjunction across the inputs and subsequently 
    applies physical axiomatic boundary negation.
    """
    def __init__(self, method: str = 'physical_kleene_dienes', gamma: float = 0.2, 
                 mode: str = 'sigmoid', slope: float = 1.0, offset: float = 0.5):
        self.or_gate = PhysicalOr(method=method, gamma=gamma, mode=mode, slope=slope, offset=offset)
        self.not_gate = PhysicalNot()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        or_res = self.or_gate(x)
        return self.not_gate(or_res)


class PhysicalXor(nnx.Module):
    """
    Parameter-free physical Exclusive OR (XOR) gate built via physical tree reduction.

    Symmetric to WeightedXor, it breaks multi-input evaluation down using a balanced 
    recursive binary tree utilizing parameter-free physical operations.
    """
    def __init__(self, num_inputs: int, method: str = 'physical_kleene_dienes', gamma: float = 0.2, 
                 mode: str = 'sigmoid', slope: float = 1.0, offset: float = 0.5):
        if num_inputs < 2:
            raise ValueError("A physical XOR gate requires at least 2 distinct inputs.")
        self.num_inputs = num_inputs
        self.method = method
        self.gamma = gamma
        self.mode = mode
        self.slope = slope
        self.offset = offset

        if num_inputs == 2:
            # Build a binary XOR element using (A OR B) AND (A NAND B) via PFL gates
            self.or_combiner = PhysicalOr(method=method, gamma=gamma, mode=mode, slope=slope, offset=offset)
            self.nand_combiner = PhysicalNand(method=method, gamma=gamma, mode=mode, slope=slope, offset=offset)
            self.and_merge = PhysicalAnd(method=method, gamma=gamma, mode=mode, slope=slope, offset=offset)
        else:
            mid = num_inputs // 2
            self.left_child = PhysicalXor(num_inputs=mid, method=method, gamma=gamma, mode=mode, slope=slope, offset=offset)
            self.right_child = PhysicalXor(num_inputs=num_inputs - mid, method=method, gamma=gamma, mode=mode, slope=slope, offset=offset)
            self.combiner = PhysicalXor(num_inputs=2, method=method, gamma=gamma, mode=mode, slope=slope, offset=offset)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.num_inputs == 2:
            # Input is of the form (..., 2, 2)
            res_or = self.or_combiner(x)
            res_nand = self.nand_combiner(x)
            combined = jnp.stack([res_or, res_nand], axis=-2)
            return self.and_merge(combined)
        else:
            mid = self.num_inputs // 2
            res_left = self.left_child(x[..., :mid, :])
            res_right = self.right_child(x[..., mid:, :])
            combined_results = jnp.stack([res_left, res_right], axis=-2)
            return self.combiner(combined_results)


# =====================================================================
# 4. COMPOUND AND DERIVED TRADITIONAL GATES (NAND, NOR, XOR, NOT)
# =====================================================================

class WeightedNand(nnx.Module):
    """
    Stateful weighted NAND gate (Alternative Denial) utilizing the Łukasiewicz core logic.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """Initializes the trainable WeightedNand gate."""
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return F.weighted_nand(x, self.weights[...], self.beta[...])


class WeightedNor(nnx.Module):
    """
    Stateful weighted NOR gate (Joint Denial) utilizing the Łukasiewicz core logic.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """Initializes the trainable WeightedNor gate."""
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return F.weighted_nor(x, self.weights[...], self.beta[...])


class WeightedXor(nnx.Module):
    """
    Trainable structural n-ary Exclusive OR (XOR) gate implemented via recursive tree reduction.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        if num_inputs < 2:
            raise ValueError("A structural XOR gate requires at least 2 distinct inputs.")
        self.num_inputs = num_inputs
        self.method = method

        if num_inputs == 2:
            if self.method == 'lukasiewicz':
                self.weights_or = nnx.Param(jnp.ones((2,)))
                self.weights_nand = nnx.Param(jnp.ones((2,)))
                self.weights_and = nnx.Param(jnp.ones((2,)))
                self.beta = nnx.Param(jnp.array(1.0))
            else:
                self.weights_or = None
        else:
            mid = num_inputs // 2
            self.left_child = WeightedXor(num_inputs=mid, rngs=rngs, method=method)
            self.right_child = WeightedXor(num_inputs=num_inputs - mid, rngs=rngs, method=method)
            self.combiner = WeightedXor(num_inputs=2, rngs=rngs, method=method)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.num_inputs == 2:
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
                raise ValueError(f"XOR method '{self.method}' is not supported inside JLNN layers.")
        else:
            mid = self.num_inputs // 2
            res_left = self.left_child(x[..., :mid, :])
            res_right = self.right_child(x[..., mid:, :])
            combined_results = jnp.stack([res_left, res_right], axis=-2)
            return self.combiner(combined_results)


class WeightedNot(nnx.Module):
    """
    Trainable weighted negation (NOT) gate with adjustable structural confidence scaling.
    """
    def __init__(self, rngs: nnx.Rngs):
        """Initializes the trainable WeightedNot inversion gate."""
        self.weight = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return F.weighted_not(x, self.weight[...])