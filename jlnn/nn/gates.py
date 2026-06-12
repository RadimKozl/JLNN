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
        weights (nnx.Param): Trainable input importance weights structured as (num_inputs,).
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
        """Initializes the parameter-free PhysicalNand gate compounding AND and NOT primitives."""
        self.and_gate = PhysicalAnd(method=method, gamma=gamma, mode=mode, slope=slope, offset=offset)
        self.not_gate = PhysicalNot()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Computes alternative denial via composition of physical gates."""
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
        """Initializes the parameter-free PhysicalNor gate compounding OR and NOT primitives."""
        self.or_gate = PhysicalOr(method=method, gamma=gamma, mode=mode, slope=slope, offset=offset)
        self.not_gate = PhysicalNot()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Computes joint denial via composition of physical gates."""
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
        """Initializes the structural PhysicalXor tree pipeline."""
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
        """Performs balanced binary structural reduction using PFL formulation rules."""
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

    This module acts as a trainable fuzzy alternative denial layer that combines feature
    importance optimization with logical inversion. In the JLNN interval framework, it
    evaluates multi-input conjunction and applies a strict bounded negation to the collective 
    result, allowing the network to suppress specific compound conditions via backpropagation.

    Attributes:
        weights (nnx.Param): Trainable input importance weights structured as (num_inputs,).
            These optimize the relative impact of individual feature streams before logical aggregation.
        beta (nnx.Param): Trainable gate activation sensitivity threshold scalar parameter (bias).
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """
        Initializes the stateful WeightedNand gate with optimization parameters.

        Args:
            num_inputs (int): Dimensionality of the incoming input feature sub-space.
            rngs (nnx.Rngs): Flax NNX random number generator collection for parameter initialization.
        """
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the weighted fuzzy NAND operation across the designated feature axis.

        Args:
            x (jnp.ndarray): Multi-variable truth interval tensor structured as 
                (..., num_inputs, 2), where the last dimension holds [Lower, Upper] bounds.

        Returns:
            jnp.ndarray: Bounded and consistency-verified truth interval structured as (..., 2).
        """
        return F.weighted_nand(x, self.weights[...], self.beta[...])


class WeightedNor(nnx.Module):
    """
    Stateful weighted NOR gate (Joint Denial) utilizing the Łukasiewicz core logic.

    This module acts as a trainable fuzzy joint denial layer that aggregates supportive 
    evidence across multiple feature streams and subsequently inverts the accumulated potential.
    It evaluates multi-input disjunction modulated by relative weights and sensitivity thresholds,
    followed by an axiomatic boundary negation to project a final strict truth interval.

    Attributes:
        weights (nnx.Param): Trainable input importance weights structured as (num_inputs,).
            These optimize the relative impact of individual feature streams before logical aggregation.
        beta (nnx.Param): Trainable gate activation sensitivity threshold scalar parameter (bias).
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """
        Initializes the stateful WeightedNor gate with optimization parameters.

        Args:
            num_inputs (int): Dimensionality of the incoming input feature sub-space.
            rngs (nnx.Rngs): Flax NNX random number generator collection for parameter initialization.
        """
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the weighted fuzzy NOR operation across the designated feature axis.

        Args:
            x (jnp.ndarray): Multi-variable truth interval tensor structured as 
                (..., num_inputs, 2), where the last dimension holds [Lower, Upper] bounds.

        Returns:
            jnp.ndarray: Bounded and consistency-verified truth interval structured as (..., 2).
        """
        return F.weighted_nor(x, self.weights[...], self.beta[...])


class WeightedXor(nnx.Module):
    """
    Trainable structural n-ary Exclusive OR (XOR) gate implemented via recursive tree reduction.

    This module evaluates multi-variable exclusive disjunction across interval-valued inputs.
    Because multi-input fuzzy XOR does not always possess associative properties under 
    parametric norms, this class structurally decomposes the reduction of an arbitrary 
    number of input signals into a balanced binary tree of 2-input XOR operators.

    For the 'lukasiewicz' method, the base 2-input operator is parameterized and stateful,
    learning distinct feature importance weights and a common activation threshold bias.
    For 'godel' and 'product' methods, the operation scales down to parameter-free analytical 
    formulations.

    Attributes:
        num_inputs (int): Dimensionality of the incoming input feature sub-space.
        method (str): Target logical framework selector ('lukasiewicz', 'godel', 'product').
        weights_or (Optional[nnx.Param]): Trainable weights for the internal OR aggregator 
            in the base 2-input Lukasiewicz operator. Shape: (2,).
        weights_nand (Optional[nnx.Param]): Trainable weights for the internal NAND aggregator 
            in the base 2-input Lukasiewicz operator. Shape: (2,).
        weights_and (Optional[nnx.Param]): Trainable weights for the final AND combiner 
            in the base 2-input Lukasiewicz operator. Shape: (2,).
        beta (Optional[nnx.Param]): Trainable shared sensitivity threshold (bias) scalar 
            for the base Lukasiewicz logic components.
        left_child (Optional[WeightedXor]): Left subtree module for inputs reduction.
        right_child (Optional[WeightedXor]): Right subtree module for inputs reduction.
        combiner (Optional[WeightedXor]): A 2-input structural gate combining subtree outputs.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """
        Initializes the stateful or structural WeightedXor gate network.

        Args:
            num_inputs (int): Dimensionality of the incoming input feature sub-space.
            rngs (nnx.Rngs): Flax NNX random number generator collection for parameter initialization.
            method (str, optional): Target fuzzy logic framework. Defaults to 'lukasiewicz'.

        Raises:
            ValueError: If `num_inputs` is less than 2, as a logical XOR operation requires 
                at least a binary interaction.
        """
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
        """
        Executes the recursive exclusive disjunction over the designated terminal feature dimension.

        Args:
            x (jnp.ndarray): Multi-variable truth interval tensor structured as 
                (..., num_inputs, 2), where the last dimension holds [Lower, Upper] bounds.

        Returns:
            jnp.ndarray: Bounded and consistency-verified truth interval structured as (..., 2).

        Raises:
            ValueError: If an unmapped or invalid logic method string is provided inside base layers.
        """
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

    This module acts as a stateful logical inversion layer within the JLNN framework. 
    It applies a fuzzy negation operator to an interval-valued truth tensor while 
    optimizing a trainable scaling weight parameter. This allows the network to learn 
    the strictness, preservation, or relaxation of structural uncertainty during 
    systemic logical inversion operations.

    Attributes:
        weight (nnx.Param): Trainable confidence scaling parameter scalar. It controls 
            the interpolation and preservation of boundary uncertainty during the 
            negation mapping.
    """
    def __init__(self, rngs: nnx.Rngs):
        """
        Initializes the trainable WeightedNot inversion gate.

        Args:
            rngs (nnx.Rngs): Flax NNX random number generator collection for parameter initialization.
        """
        self.weight = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the parameterized logical negation over the input interval.

        Args:
            x (jnp.ndarray): Input truth interval tensor structured as (..., 2),
                where the last dimension holds [Lower, Upper] bounds.

        Returns:
            jnp.ndarray: Bounded and consistency-verified inverted truth interval 
                structured as (..., 2).
        """
        return F.weighted_not(x, self.weight[...])