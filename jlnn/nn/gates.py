#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from flax import nnx
from jlnn.nn import functional as F

class WeightedOr(nnx.Module):
    """
    Trainable or self-adaptive structural OR gate supporting multiple fuzzy logical semantics.

    In the JLNN framework, this module acts as a stateful neural layer aggregating 
    truth intervals across multiple input streams. For traditional parametric semantics, 
    it optimizes relative feature importance via weights and shifts the collective 
    activation threshold via a learnable beta parameter. For Physical Fuzzy Logic (PFL), 
    it performs parameter-free, entropy-driven space-warped disjunction.

    Attributes:
        method (str): Target logical framework selector. Supported types: 'lukasiewicz', 
            'godel', 'product', 'drastic', and 'physical_kleene_dienes'.
        weights (Optional[nnx.Param]): Trainable input importance weights structured 
            as (num_inputs,). None if the selected method is physical.
        beta (Optional[nnx.Param]): Trainable gate sensitivity threshold scalar parameter (bias). 
            None if the selected method is physical.
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
        
        # Physical methods (PFL) are inherently parameter-free.
        if not self.method.startswith('physical_'):
            self.weights = nnx.Param(jnp.ones((num_inputs,)))
            self.beta = nnx.Param(jnp.array(1.0))
        else:
            self.weights = None
            self.beta = None

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
    Trainable or self-adaptive structural AND gate supporting multiple fuzzy logical semantics.

    Acts as a stateful neural intersection layer evaluating systemic conjunction across 
    multiple truth interval inputs. Depending on the chosen semantics, it either minimizes 
    joint evidence through parameterized linear optimizations (weights/beta) or dynamically 
    warps the underlying truth space according to Shannon entropy indicators.

    Attributes:
        method (str): Target logical framework selector. Supported types: 'lukasiewicz', 
            'godel', 'product', 'drastic', and 'physical_kleene_dienes'.
        weights (Optional[nnx.Param]): Trainable importance weights scaling individual 
            inputs. None if the method is physical.
        beta (Optional[nnx.Param]): Trainable gate activation threshold bias parameter. 
            None if the method is physical.
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
        
        if not self.method.startswith('physical_'):
            self.weights = nnx.Param(jnp.ones((num_inputs,)))
            self.beta = nnx.Param(jnp.array(1.0))
        else:
            self.weights = None
            self.beta = None

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
    Non-trainable pure stateless bulk AND reduction hradlo (gate).

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
            ValueError: If the method is unrecognized or expects parametric weights.
        """
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
    Non-trainable pure stateless bulk OR reduction hradlo (gate).

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
            ValueError: If the method is unrecognized or expects parametric weights.
        """
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
    Stateful weighted NAND gate (Alternative Denial) utilizing the Łukasiewicz core logic.

    Computes the weighted alternative denial by first evaluating a stateful 
    Łukasiewicz conjunction and subsequently performing axiomatic boundary negation.

    Attributes:
        weights (nnx.Param): Trainable feature optimization weights structured as (num_inputs,).
        beta (nnx.Param): Trainable threshold activation parameter scaling conjunction strictness.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """Initializes the trainable WeightedNand gate."""
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the weighted NAND operation over truth intervals.

        Args:
            x (jnp.ndarray): Interval tensor structured as (..., num_inputs, 2).

        Returns:
            jnp.ndarray: Negated conjunctive truth interval structured as (..., 2).
        """
        return F.weighted_nand(x, self.weights[...], self.beta[...])


class WeightedNor(nnx.Module):
    """
    Stateful weighted NOR gate (Joint Denial) utilizing the Łukasiewicz core logic.

    Computes the weighted joint denial by first evaluating a stateful 
    Łukasiewicz disjunction and subsequently performing axiomatic boundary negation.

    Attributes:
        weights (nnx.Param): Trainable feature optimization weights structured as (num_inputs,).
        beta (nnx.Param): Trainable threshold saturation parameter scaling disjunction limits.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """Initializes the trainable WeightedNor gate."""
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the weighted NOR operation over truth intervals.

        Args:
            x (jnp.ndarray): Interval tensor structured as (..., num_inputs, 2).

        Returns:
            jnp.ndarray: Negated disjunctive truth interval structured as (..., 2).
        """
        return F.weighted_nor(x, self.weights[...], self.beta[...])


class WeightedXor(nnx.Module):
    """
    Trainable or structural n-ary Exclusive OR (XOR) gate implemented via recursive tree reduction.

    Fuzzy XOR operations are inherently binary. To evaluate multi-input signals uniformly 
    without causing analytical vanishing gradients, this class constructs a balanced recursive 
    binary tree. Leaves handle pair-wise sub-expressions, which are bubbled up and merged 
    by an internal combiner gate. Supports Łukasiewicz, Gödel, and Product logic variants.

    Attributes:
        num_inputs (int): Number of inputs processed by this specific node level.
        method (str): Selected structural semantics ('lukasiewicz', 'godel', 'product').
        weights_or (Optional[nnx.Param]): Trainable parameters for the OR sub-gate (Łukasiewicz binary base only).
        weights_nand (Optional[nnx.Param]): Trainable parameters for the NAND sub-gate (Łukasiewicz binary base only).
        weights_and (Optional[nnx.Param]): Trainable parameters for the final merging AND gate (Łukasiewicz only).
        beta (Optional[nnx.Param]): Threshold gating parameter (Łukasiewicz only).
        left_child (Optional[WeightedXor]): Left sub-tree node for rekurzivní division.
        right_child (Optional[WeightedXor]): Right sub-tree node for rekurzivní division.
        combiner (Optional[WeightedXor]): Terminal binary combiner node merging child operations.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """
        Builds the rekurzivní binary structural tree for the XOR evaluation.

        Args:
            num_inputs (int): Total number of input signals entering this gate layer.
            rngs (nnx.Rngs): Flax NNX random number generator collection.
            method (str, optional): Target framework semantics. Defaults to 'lukasiewicz'.

        Raises:
            ValueError: If num_inputs is less than 2.
        """
        if num_inputs < 2:
            raise ValueError("A structural XOR gate requires at least 2 distinct inputs.")
        self.num_inputs = num_inputs
        self.method = method

        if num_inputs == 2:
            # Base Case: Initialize direct binary processing operators
            if self.method == 'lukasiewicz':
                self.weights_or = nnx.Param(jnp.ones((2,)))
                self.weights_nand = nnx.Param(jnp.ones((2,)))
                self.weights_and = nnx.Param(jnp.ones((2,)))
                self.beta = nnx.Param(jnp.array(1.0))
            else:
                # Gödel and Product XOR use analytical pure functions from functional.py
                self.weights_or = None
        else:
            # Recursive Case: Subdivide inputs and allocate tree modules
            mid = num_inputs // 2
            self.left_child = WeightedXor(num_inputs=mid, rngs=rngs, method=method)
            self.right_child = WeightedXor(num_inputs=num_inputs - mid, rngs=rngs, method=method)
            self.combiner = WeightedXor(num_inputs=2, rngs=rngs, method=method)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluates the exclusive disjunction by rekurzivně traversing the allocated tree structure.

        Args:
            x (jnp.ndarray): Truth interval tensor structured as (..., num_inputs, 2).

        Returns:
            jnp.ndarray: Final exclusive disjunction truth interval structured as (..., 2).

        Raises:
            ValueError: If the selected semantic method is unsupported.
        """
        if self.num_inputs == 2:
            # Terminal Leaf Node: Compute standard binary XOR operations
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


class WeightedImplication(nnx.Module):
    """
    Trainable or self-adaptive directional implication gate (A -> B).

    Acts as a stateful functional rule gateway within the network architecture, mapping 
    causal logic relationships between an antecedent (premise) and a consequent (conclusion). 
    Supports standard logical frameworks and advanced space-curved Physical Fuzzy Logic (PFL) rules.

    Attributes:
        method (str): Chosen architectural semantics (e.g., 'lukasiewicz', 'goguen', 
            'physical_kleene_dienes', etc.).
        weights (Optional[nnx.Param]): Trainable implication optimization weights scaling 
            rule importance as [Weight_A, Weight_B]. None for physical systems.
        beta (Optional[nnx.Param]): Trainable gate sensitivity bias parameter. None for physical systems.
    """
    def __init__(self, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """Initializes the directional WeightedImplication gate layer."""
        self.method = method
        
        if not self.method.startswith('physical_'):
            self.weights = nnx.Param(jnp.ones((2,)))
            self.beta = nnx.Param(jnp.array(1.0))
        else:
            self.weights = None
            self.beta = None

    def __call__(self, int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the structural logical implication between antecedent and consequent intervals.

        Args:
            int_a (jnp.ndarray): Antecedent truth interval tensor structured as (..., 2).
            int_b (jnp.ndarray): Consequent truth interval tensor structured as (..., 2).

        Returns:
            jnp.ndarray: Consistency-verified rule validity truth interval structured as (..., 2).
        """
        if self.weights is not None:
            w = self.weights[...]
            b = self.beta[...]
        else:
            # Neutral mask bypass to keep function signatures uniform across functional backends
            w = jnp.ones((2,))
            b = jnp.array(1.0)

        return F.weighted_implication(
            int_a, int_b, w, b, self.method
        )


class WeightedNot(nnx.Module):
    """
    Trainable weighted negation (NOT) gate with adjustable structural confidence scaling.

    Allows the model to learn the strictness or preservation of structural uncertainty 
    during systemic logical inversion operations.

    Attributes:
        weight (nnx.Param): Trainable interpolation confidence parameter scalar.
    """
    def __init__(self, rngs: nnx.Rngs):
        """Initializes the trainable WeightedNot inversion gate."""
        self.weight = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the parameterized logical negation over the input interval.

        Args:
            x (jnp.ndarray): Input truth interval tensor structured as (..., 2).

        Returns:
            jnp.ndarray: Confidence-scaled inverted truth interval structured as (..., 2).
        """
        return F.weighted_not(x, self.weight[...])